include("tigerV4-world.jl");

############################################## STATE

# Returns the complete state space of a POMDP
POMDPs.states(decision_problem::TigerDp) = decision_problem.state_space;
POMDPs.n_states(decision_problem::TigerDp) = length(decision_problem.state_space)
POMDPs.state_index(decision_problem::TigerDp, state_at_door::StateAtDoor) = get_state_index(decision_problem, state_at_door)


############################################## ACTION

# Returns the entire action space of a POMDP
POMDPs.actions(decision_problem::TigerDp) = decision_problem.action_space
POMDPs.actions(decision_problem::TigerDp, state::StateAtDoor) = decision_problem.action_space     # every state has the same actions
POMDPs.n_actions(decision_problem::TigerDp) = length(decision_problem.action_space)
POMDPs.action_index(decision_problem::TigerDp, action::Symbol) = get_action_index(decision_problem, action)


############################################## TRANSITION

# Return the transition distribution from the current state-action pair
POMDPs.transition(decision_problem::TigerDp, cat_state::StateAtDoor, action::Symbol) = get_state_prime(decision_problem, cat_state, action)


############################################## OBSERVATION

# Return the entire observation space
POMDPs.observations(decision_problem::TigerDp) = get_observation_space();
POMDPs.observations(decision_problem::TigerDp, state_at_door::StateAtDoor) = get_observation_space();
POMDPs.n_observations(decision_problem::TigerDp) = length(get_observation_space());
POMDPs.obs_index(decision_problem::TigerDp, observation_at_door::ObservationAtDoor) = get_obs_index(decision_problem, observation_at_door)

# Return the observation distribution for a state (this method can only be implemented when the observation does not depend on the action)
POMDPs.observation(decision_problem::TigerDp, state_at_door::StateAtDoor) = get_current_observation(decision_problem, state_at_door)

############################################## REWARD

# Return the immediate reward for the s-a pair
POMDPs.reward(decision_problem::TigerDp, door_state::StateAtDoor, action::Symbol) = get_reward(decision_problem, door_state, action)


############################################## BELIEF

# Returns an initial belief for the pomdp
POMDPs.initial_state_distribution(decision_problem::TigerDp) = StatePriorDistribution();