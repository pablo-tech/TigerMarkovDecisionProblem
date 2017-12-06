### REFERENCES
# Intro to the tiger problem https://www.techfak.uni-bielefeld.de/~skopp/Lehre/STdKI_SS10/POMDP_tutorial.pdf
# Beliefs, Distributions, Model, Policies, Simulators https://github.com/JuliaPOMDP/POMDPToolbox.jl
# POMDPs.jl API http://juliapomdp.github.io/POMDPs.jl/latest/api/#POMDPs.state_index
# Tutorial Tiger POMDP http://nbviewer.jupyter.org/github/sisl/POMDPs.jl/blob/master/examples/Tiger.ipynb
# Explicit POMDP http://juliapomdp.github.io/POMDPs.jl/latest/explicit/
# Tiger problem PDF https://www.cs.rutgers.edu/%7Emlittman/papers/aij98-pomdp.pdf
# Tiger problem PPT https://www.techfak.uni-bielefeld.de/~skopp/Lehre/STdKI_SS10/POMDP_tutorial.pdf

### POMDPS.jl
# The following may be performed prior and separately by own JuliaCommand.jl
importall POMDPs
POMDPs.add("QMDP")                  # Design Under Uncertainty 6.4.1
POMDPs.add("SARSOP")
using QMDP
using SARSOP
using POMDPToolbox
using POMDPModels


### PROBLEM
# In the tiger POMDP, the agent is tasked with escaping from a room. There are two doors leading out of the room.
# Behind one of the doors is a tiger, and behind the other is sweet, sweet freedom.
# If the agent opens the door and finds the tiger, it gets eaten (and receives a reward of -100).
# If the agent opens the other door, it escapes and receives a reward of 10.
# The agent can also listen. Listening gives a noisy measurement of which door the tiger is hiding behind.
# Listening gives the agent the correct location of the tiger 85% of the time.
# The agent receives a reward of -1 for listening.


### PARTIALLY OBSERVABLE MARKOV DECISION PROBLEM
# definition:
# S: State
# A: Action
# Omega: Observation space
# O: Observation function
# T: Transition
# R: Reward

############################################## WORLD

### WORLD: StateAtDoor
# hidden state behind each door
struct StateAtDoor
    door_name::Symbol
    cat_is_here::Bool
end

### WORLD: ObservationAtDoor
# direct observation at each door
struct ObservationAtDoor
    door_name::Symbol
    cat_heard_here::Bool
end

### WORLD: class
type TigerDp <: POMDP{StateAtDoor, Symbol, ObservationAtDoor}          # inherits typed POMDP{State, Action, Observation}
    reward_for_listening::Float64                       # -1.0
    reward_for_killed_by_tiger::Float64                 # -100
    reward_for_evading_tiger::Float64                   # 10
    probability_of_listening_correctly::Float64         # 0.85
    discount_factor::Float64                            # 0.95
    door_list::Vector{Symbol}
    state_space::Vector{StateAtDoor}
    action_space::Vector{Symbol}
    observation_space::Vector{ObservationAtDoor}
end

# modify the state space at termination of the game
function initialize_randomnized_problem(decision_problem::TigerDp)
    new_state_space = state_random_space(decision_problem.door_list)
    println("***RESTART*** Randomnized cat location!!! ", new_state_space)
    decision_problem.state_space = new_state_space
    # return decision_problem
end


############################################## STATE
# http://juliapomdp.github.io/POMDPs.jl/latest/api/#POMDPs.states
# the tiger is either behind the left door or behind the right door.
# Our state space is simply an array of the states in the problem.

# create a random state, with empty name
function state_random_door()
    return state_random_door("")
end

function is_cat_on(state_at_doors)
    for door in state_at_doors
        if door.cat_is_here
            return true
        end
    end
    return false
end

# pick one of the doors to hide the cat
function hide_cat_where(door_list)
    return rand(1:length(door_list))
end

# state space: random set of states
function state_random_space(door_list)
    hideout = hide_cat_where(door_list)
    state_at_doors = StateAtDoor[]
    door_count = 1
    for door_name in door_list
        if door_count==hideout
            door_has_cat = true
        else
            door_has_cat = false
        end
        new_random_door = StateAtDoor(door_name, door_has_cat)
        push!(state_at_doors, new_random_door)
        door_count+=1
    end
    return state_at_doors
end

# prior distribution of state
function StatePriorDistribution(decision_problem::TigerDp)
    distribution_length = length(decision_problem.state_space)
    uniform_proability = 1/distribution_length
    probability_vector = Vector{Float64}(distribution_length)
    for i = 1:distribution_length
        probability_vector[i]=uniform_proability
    end
    sparse_cat = SparseCat(decision_problem.state_space, probability_vector)
    println("SPARSE_CAT ", sparse_cat)
    return sparse_cat
end

# prior distribution of state
function StatePosteriorDistribution(decision_problem::TigerDp, cat_state::StateAtDoor, probability::Float64)
    probability_remainder = 1 - probability
    others_count = length(decision_problem.state_space) - 1
    others_probability_each = probability_remainder/others_count

    distribution_length = length(decision_problem.state_space)
    probability_vector = Vector{Float64}(distribution_length)

    for i = 1:distribution_length
        if cat_state.door_name==decision_problem.state_space[i].door_name
           probability_vector[i]=probability               # main recipient
        else
           probability_vector[i]=others_probability_each
        end
    end
    return SparseCat(decision_problem.state_space, probability_vector)
end

# state index
function get_state_index(decision_problem::TigerDp, state_at_door::StateAtDoor)
    for i = 1:length(decision_problem.state_space)
        if state_at_door.door_name==decision_problem.state_space[i].door_name
            # println(i, " FOUND: ", state_at_door, " STATE INDEX TO? ", decision_problem.state_space[i])
            return i
        end
    end
    # println("STATE NOT FOUND IN STATE SPACE: ", cat_state, "\t", state_space)
    return 1
end



############################################## ACTION
# There are three possible actions our agent can take: open the left door, open the right door, and listen.

# action space
function get_action_index(decision_problem::TigerDp, action::Symbol)
    for i = 1:length(decision_problem.action_space)
        if action==decision_problem.action_space[i]
            # println(i, " FOUND: ", action, " ACTION INDEX TO? ", decision_problem.action_space[i])
            return i
        end
    end
    println("ACTION NOT FOUND IN ACTION SPACE: ", action)
    return 1
end


############################################## OBSERVATION
# The observation space looks similar to the state space
# State is the truth about our system. observation is potentially false information received about the state
# the agent either hears the tiger behind the left door, or behind the right door, or not at all


# observation index
function get_obs_index(decision_problem::TigerDp, observation_at_door::ObservationAtDoor)
    for i = 1:length(decision_problem.observation_space)
        if observation_at_door.door_name==decision_problem.observation_space[i].door_name
            println(i, " FOUND: ", observation_at_door, " OBSERVATION INDEX TO? ", decision_problem.observation_space[i])
            return i
        end
    end
    println("STATE NOT FOUND IN STATE SPACE: ", observation_at_door, "\t", decision_problem.state_space)
    return 1
end

# prior distribution of observation
function ObservationPriorDistribution(decision_problem::TigerDp)
    distribution_length = length(decision_problem.observation_space)
    uniform_proability = 1/distribution_length
    probability_vector = Vector{Float64}(distribution_length)
    for i = 1:distribution_length
        probability_vector[i]=uniform_proability
    end
    return SparseCat(decision_problem.observation_space, probability_vector)
end

# prior distribution of state
function ObservationPosteriorDistribution(decision_problem::TigerDp, cat_observation::ObservationAtDoor, probability::Float64)
    probability_remainder = 1 - probability
    others_count = length(decision_problem.state_space) - 1
    others_probability_each = probability_remainder/others_count

    distribution_length = length(decision_problem.state_space)
    probability_vector = Vector{Float64}(distribution_length)

    for i = 1:distribution_length
        if cat_observation.door_name==decision_problem.observation_space[i].door_name
           probability_vector[i]=probability               # main recipient
        else
           probability_vector[i]=others_probability_each
        end
    end
    return SparseCat(decision_problem.observation_space, probability_vector)
end

# observation function
# The observation model captures the uncertaintiy in the agent's listening ability.
# When we listen, we receive a noisy measurement of the tiger's location.
# Return the observation distribution for a state (this method can only be implemented when the observation does not depend on the action)
function get_current_observation(decision_problem::TigerDp, state_at_door::StateAtDoor)
    # obtain correct observation a % of the time
     if state_at_door.cat_is_here
        # noise impedes listenting
        probability = decision_problem.probability_of_listening_correctly
     else
         probability = 1.0 - decision_problem.probability_of_listening_correctly
     end
     door_observation = ObservationAtDoor(state_at_door.door_name, state_at_door.cat_is_here)
     return ObservationPosteriorDistribution(door_observation, probability)
 end;

# TODO: when is this called???
function get_current_observation(decision_problem::TigerDp, state_at_door::SparseCat)
     println("STATE_AT_DOOR ", state_at_door)
    # TODO: implement, how?
     return SparseCat(decision_problem.observation_space, [0.3,0.7])
end


############################################## REWARD

### REWARD
# -1 for listening at the door
# -100 for encountering the tiger
# +10 for escaping

function get_reward(decision_problem::TigerDp, door_state::StateAtDoor, user_action::Symbol)
    reward = 0.0
    # listening
    if user_action==:listen_to_doors
        # println("-REWARD! Listening to door!!!")
        reward+=decision_problem.reward_for_listening
    end
    # opening
    if user_action==:open_left_door || user_action==:open_right_door
        if door_state.cat_is_here
            println("---REWARD! Killed by cat!!!")
            reward += decision_problem.reward_for_killed_by_tiger
        else
            println("++REWARD! Evaded the cat!!!")
            reward += decision_problem.reward_for_evading_tiger
        end
    end
    return reward
end

# TODO: when is this called???
function get_reward(decision_problem::TigerDp, door_state::SparseCat, user_action::Symbol)
    println("???WHAT_REWAWRD???")
    # TODO: implement, how?
    return 0.0
end

############################################## TRANSITION

### TRANSITION MODEL: across episodes
# The location of the tiger doesn't change when the agent listens.
# After the agent opens the door, the game episode terminates, and no further reward is earned. (***)

# Across episode, resets the problem after opening door; does nothing while playing/listening
function get_state_prime(decision_problem::TigerDp, door_state::StateAtDoor, user_action::Symbol)
    if user_action==:open_left_door || user_action==:open_right_door
        if door_state.cat_is_here
            println("LILY KILLED BY CAT BEHIND DOOR: ", door_state)
            # game termination, back to prior: door was opened
            initialize_randomnized_problem(decision_problem)
            return StatePriorDistribution(decision_problem)
        end
    end
    # if instead of opening she listened
    if door_state.cat_is_here
        println("HEARD IT! Now sure the cat is here!!! ", door_state)
        return StatePosteriorDistribution(decision_problem, door_state, 1.0)
    end
    return StatePosteriorDistribution(decision_problem, door_state, 0.0)
end;


############################################## BELIEF
# In general terms, a belief is something that is mapped to an action using a POMDP policy
# Solvers perform the belief updating.  All you need to do is use/define distribution over states.

# Return the next state sp, observation o and reward for taking action a in current state s
# (sp, o, r) = POMDPs.generate_sor(pomcp.problem, s, a, sol.rng)
function get_state_observation_reward(decision_problem::TigerDp, state_at_door::StateAtDoor, user_action::Symbol, random_number_generator::AbstractRNG)
    println("^^^^^get_state_observation_reward ", "s=",state_at_door, " a=", user_action)
    state_at_door = get_state_prime(decision_problem, state_at_door, user_action)
    action_reward = get_reward(decision_problem, state_at_door, user_action)
    observation_at_door = get_current_observation(decision_problem, state_at_door)
    (state_at_door, observation_at_door, action_reward)
end
