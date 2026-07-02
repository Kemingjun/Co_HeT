import torch
from torch import nn
import math
from nets.graph_encoder import GraphAttentionEncoder
from nets.echo_task_encoder import EchoTaskEncoder
from nets.robot_encoder import RobotAttentionEncoder, RobotTypePromptEncoder
from nets.task_robot_attention import TaskRobotCrossAttention
from torch.utils.checkpoint import checkpoint
from torch.nn import DataParallel
from utils.functions import sample_many
from problems.hrsp.paramet_hrsp import paramet_hrsp
import torch.nn.functional as F


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 use_echo_encoder=True,
                 use_pfca_context=True,
                 echo_edge_dim=128):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0



        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits


        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.use_echo_encoder = use_echo_encoder
        self.use_pfca_context = use_pfca_context



        self.norm_query_task = nn.LayerNorm(embedding_dim)

        # self.norm_glimpse_task = nn.LayerNorm(embedding_dim)


        self.norm_query_robot = nn.LayerNorm(embedding_dim)

        # self.norm_glimpse_robot = nn.LayerNorm(embedding_dim)


        node_dim = 3 + paramet_hrsp.ROBOT_TYPE_NUM

        # [x, y, cur_time, type(one_hot)]
        robot_dim = 3

        self.robot_init_linear = nn.Linear(robot_dim, embedding_dim, bias=False)



        if self.use_echo_encoder:
            self.graph_embedder = EchoTaskEncoder(
                node_dim=node_dim,
                embed_dim=embedding_dim,
                n_heads=n_heads,
                n_layers=1,
                normalization=normalization,
                edge_dim=echo_edge_dim
            )
        else:
            self.graph_init_linear = nn.Linear(node_dim, embedding_dim, bias=False)
            self.graph_embedder = GraphAttentionEncoder(
                n_heads=n_heads,
                embed_dim=embedding_dim,
                n_layers=1,
                normalization=normalization
            )

        self.robot_embedder = RobotAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=1,
            normalization='layer'
        )


        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_robot_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_task_query = nn.Linear(embedding_dim * 2, embedding_dim, bias=False)

        assert embedding_dim % n_heads == 0
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_out_robot = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.project_robot_query = nn.Linear(3 * embedding_dim, embedding_dim, bias=False)
        self.project_context = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        self.project_node_mask = nn.Linear(embedding_dim + 1, embedding_dim, bias=False)


        self.empty_robots_embedding = nn.Parameter(torch.empty(embedding_dim))
        nn.init.xavier_uniform_(self.empty_robots_embedding.unsqueeze(0))
        self.empty_coalition_embedding = nn.Parameter(torch.empty(embedding_dim))
        nn.init.xavier_uniform_(self.empty_coalition_embedding.unsqueeze(0))

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        node_embeddings, graph_embedding = self._encode_tasks(input)


        _log_p, pi, cost, distance, tardiness = self._inner(input, node_embeddings)

        # cost, mask = self.problem.get_costs(input, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = _log_p.sum(1)
        if return_pi:
            return cost, ll, pi

        return cost, ll, distance, tardiness

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)



    def _calc_log_likelihood(self, _log_p, a):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        # if mask is not None:
        #     log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        # return log_p.mean(1)
        return log_p.sum(1)

    def graph_init_embed(self, input):
        max_deadline = input['deadline'].max()
        paramet_hrsp.time_norm = math.ceil(max_deadline)
        node_info = torch.cat((
            input['source'],
            input['deadline'][:, :, None] / paramet_hrsp.time_norm,
            input['operation_time']
            ), dim=-1)

        if self.use_echo_encoder:
            return node_info
        return self.graph_init_linear(node_info)

    def _encode_tasks(self, input):
        graph_input = self.graph_init_embed(input)
        if self.checkpoint_encoder and self.training:
            if self.use_echo_encoder:
                return checkpoint(self.graph_embedder, graph_input, input['source'], use_reentrant=False)
            return checkpoint(self.graph_embedder, graph_input, use_reentrant=False)
        if self.use_echo_encoder:
            return self.graph_embedder(graph_input, input['source'])
        return self.graph_embedder(graph_input)

    def robot_init_embed(self, state):
        assert paramet_hrsp.time_norm is not None, 'paramet_hrsp.time_norm is None'
        robot_init_info = torch.cat((
            state.cur_coord,
            state.cur_time.unsqueeze(-1) / paramet_hrsp.time_norm,
            # paramet_hrsp.robot_one_hot.to(state.cur_coord.device).unsqueeze(0).expand(state.cur_coord.size(0), -1, -1)
        ), -1)
        return self.robot_init_linear(robot_init_info)

    def get_robot_mask(self, input):
        required_robot = input['required_robot']

        required_status_for_each_robot = required_robot[:, :, paramet_hrsp.robot_type_indices]

        # The mask should be True where the robot is NOT required (i.e., required_status is 0)
        robot_type_mask = (required_status_for_each_robot == 0)

        return robot_type_mask


    def _inner(self, input, node_embeddings):

        outputs = []
        sequences = []

        state = self.problem.make_state(input)

        self.batch_idx = torch.arange(node_embeddings.size(0), device=node_embeddings.device)
        prev_coalition_embedding = self.empty_coalition_embedding.expand(node_embeddings.size(0), -1)

        # Perform decoding steps
        i = 0
        while not state.all_finished():

            robot_embeddings, robot_embd_mean = self.robot_embedder(self.robot_init_embed(state))
            task_embeddings_for_step = self._apply_pfca_context(node_embeddings, prev_coalition_embedding)

            mask = state.get_mask()
            node_embeddings_with_mask = self.project_node_mask(torch.cat((task_embeddings_for_step, mask.transpose(1, 2)), dim=-1))

            env_info = torch.concat([node_embeddings_with_mask.mean(1), robot_embd_mean], dim=1)
            env_context = self.project_context(env_info)
            selected_robot_one_hot, robot_selected, robot_log_p, selected_robots_embedding_list = self.robot_decision(robot_embeddings, env_context)
            _robot_log_p = robot_log_p.squeeze(1)

            selected_robot_log_p = torch.gather(
                _robot_log_p,
                dim=1,
                index=robot_selected
            )  # (1024, 2)

            selected_robots_embedding = torch.cat(selected_robots_embedding_list, dim=1)
            selected_robots_embedding_mean = selected_robots_embedding.mean(dim=1)

            task_log_p, mask = self._get_log_p_task(task_embeddings_for_step, task_embeddings_for_step.mean(1), selected_robots_embedding_mean,
                                                    state)

            selected_task = self._select_node(task_log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension

            _task_log_p = task_log_p.squeeze(1)
            _selected_task = selected_task.unsqueeze(-1)
            selected_task_log_p = torch.gather(
                _task_log_p,
                dim=1,
                index=_selected_task
            )

            sum_selected_robot_log_p = selected_robot_log_p.sum(dim=1).unsqueeze(1)  # (1024,)

            log_p = selected_task_log_p + sum_selected_robot_log_p

            state = state.update(selected_task, selected_robot_one_hot)
            prev_coalition_embedding = selected_robots_embedding_mean

            action = torch.cat([_selected_task, robot_selected], dim=-1)

            outputs.append(log_p)
            sequences.append(action)

            i += 1
        sum_length = state.length.sum(dim=1)
        cost = sum_length.unsqueeze(-1) * paramet_hrsp.WEIGHT + state.tardiness * (1 - paramet_hrsp.WEIGHT)

        return torch.stack(outputs, 1), torch.stack(sequences, 1), cost, sum_length.unsqueeze(-1), state.tardiness

    def _apply_pfca_context(self, node_embeddings, prev_coalition_embedding):
        if not self.use_pfca_context:
            return node_embeddings
        scores = torch.matmul(
            node_embeddings,
            prev_coalition_embedding.unsqueeze(-1)
        ) / math.sqrt(node_embeddings.size(-1))
        weights = torch.softmax(scores, dim=1)
        return node_embeddings + weights * prev_coalition_embedding.unsqueeze(1)


    def sample_many(self, _input, batch_rep=1, iter_rep=1):
        """
        :param _input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        graph_embeddings = self._encode_tasks(_input)
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            (_input, graph_embeddings[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _select_robot(self, probs):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)


        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232

        else:
            assert False, "Unknown decode type"
        return selected

    def _get_log_p_task(self, node_robot_embeddings, node_robot_mean_embeddings, robot_embd_sum, state, normalize=True):
        query_info = torch.concat([node_robot_mean_embeddings, robot_embd_sum], dim=-1)
        query = self.project_task_query(query_info)[:, None, :]
        query = self.norm_query_task(query + node_robot_mean_embeddings.unsqueeze(1))

        glimpse_K, glimpse_V, logit_K = \
            self.project_node_embeddings(node_robot_embeddings[:, None, :, :]).chunk(3, dim=-1)

        glimpse_K = self._make_heads(glimpse_K, 1)
        glimpse_V = self._make_heads(glimpse_V, 1)
        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        # self.last_glimpse = glimpse
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def robot_decision(self, robot_embeddings, env_context):

        # selected_task_node_emb = node_embeddings[self.batch_idx, selected_task]


        selected_robots_embedding_list = []


        selected_robot_one_hot = None
        robot_log_p = None
        robot_selected = None   # [batch_size, 1234]
        for robot_type_index in range(paramet_hrsp.ROBOT_TYPE_NUM):
            robot_type_log_p, robot_type_embeddings = self._get_log_p_robot(robot_embeddings, env_context, robot_type_index, selected_robots_embedding_list)
            selected_type_robot = self._select_robot(robot_type_log_p.exp()[:, 0, :])


            selected_robot_type_embedding = robot_type_embeddings[self.batch_idx, selected_type_robot]

            selected_robots_embedding_list.append(selected_robot_type_embedding[:, None, :])

            selected_type_robot_one_hot = F.one_hot(selected_type_robot,
                                                    num_classes=paramet_hrsp.ROBOT_NUM_LIST[robot_type_index]).float()
            if selected_robot_one_hot is None:
                selected_robot_one_hot = selected_type_robot_one_hot
            else:
                selected_robot_one_hot = torch.cat([selected_robot_one_hot, selected_type_robot_one_hot], -1)

            if robot_log_p is None:
                robot_log_p = robot_type_log_p
            else:
                robot_log_p = torch.cat([robot_log_p, robot_type_log_p], -1)

            if robot_selected is None:
                robot_selected = selected_type_robot.unsqueeze(-1)
            else:
                offset_selected_type_robot = selected_type_robot + paramet_hrsp.ROBOT_NUM_LIST[:robot_type_index].sum()
                robot_selected = torch.cat([robot_selected, offset_selected_type_robot.unsqueeze(-1)], -1)
        return selected_robot_one_hot, robot_selected, robot_log_p, selected_robots_embedding_list



    def _get_log_p_robot(self, robot_embeddings, env_context, robot_type_index, selected_robots_embedding_list, normalize=True):

        start_index = paramet_hrsp.ROBOT_NUM_LIST[:robot_type_index].sum()
        end_index = start_index + paramet_hrsp.ROBOT_NUM_LIST[robot_type_index]

        robot_type_embedding = robot_embeddings[:, start_index: end_index, :]

        if len(selected_robots_embedding_list) == 0:
            selected_robots_embedding_mean = self.empty_robots_embedding.expand(robot_type_embedding.size(0), -1)
        else:
            selected_robots_embedding = torch.cat(selected_robots_embedding_list, dim=1)
            selected_robots_embedding_mean = selected_robots_embedding.mean(dim=1)

        context_info = torch.cat((env_context, robot_type_embedding.mean(dim=1), selected_robots_embedding_mean), dim=-1)

        query = self.project_robot_query(context_info).unsqueeze(1)

        query = self.norm_query_robot(query + robot_type_embedding.mean(dim=1).unsqueeze(1))


        glimpse_K_, glimpse_V_, logit_K_ = self.project_robot_embeddings(robot_type_embedding[:, None, :, :]).chunk(3, dim=-1)

        glimpse_K = self._make_heads(glimpse_K_, 1)
        glimpse_V = self._make_heads(glimpse_V_, 1)
        logit_K = logit_K_.contiguous()


        log_p, glimpse = self._one_to_many_logits_robot(query, glimpse_K, glimpse_V, logit_K)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, robot_type_embedding


    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # glimpse = self.norm_glimpse_task(glimpse + query.unsqueeze(1))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'

        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _one_to_many_logits_robot(self, query, glimpse_K, glimpse_V, logit_K):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        # if self.mask_inner:
        #     assert self.mask_logits, "Cannot mask inner without masking logits"
        #     compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf



        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out_robot(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # glimpse = self.norm_glimpse_robot(glimpse + query.unsqueeze(1))
        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'

        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping


        return logits, glimpse.squeeze(-2)



    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )


