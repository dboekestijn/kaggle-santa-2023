core_count = 16;
max_sim_depth = 1e3;
state_size = 12;
num_moves = 16;
num_shuffles = 10;

solution_state = randi(6, 1, state_size);
moves = zeros(num_moves, state_size);
for i=1:num_moves
    moves(i, :) = randperm(state_size);
end

from_state = solution_state;
for i=1:num_shuffles
    from_state = from_state(moves(randi(num_moves), :));
end

simulator = MCTSimulator(core_count, max_sim_depth, solution_state, moves);
shortest_move_path = simulator.simulateAll(from_state, core_count);
disp(shortest_move_path)
disp(size(shortest_move_path, 1))