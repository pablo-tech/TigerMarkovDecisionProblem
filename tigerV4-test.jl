include("tigerV4-world.jl");
include("tigerV4-POMDP.jl");


############################################# WORLD: INSTANCE

### WORLD: state
door_list = [:left, :right]

first_state_space = state_random_space(door_list)

# WORLD: action
action_space = [:open_left_door, :open_right_door, :listen_to_doors]

# WORLD: observation
# initially no observation
observation_space = [ObservationAtDoor(:left, false), ObservationAtDoor(:right, false)]

### WORLD: object
# create the world, default constructor necessary by simmulator
function TigerDp()
    TigerDp(-1.0, -100.0, 10.0, 0.85, 0.95, door_list, first_state_space, action_space, observation_space)
end

cat_dp = TigerDp()



############################################# TEST

### TEST: WORLD
test_state=StateAtDoor(:left,true)
test_obs=ObservationAtDoor(:right,false)
println("STATE: ", test_state)
println("OBSERVATION: ", test_obs)
println("WORLD: ", cat_dp)

### TEST: STATE
println("CAT WORLD PRIOR STATE PDF: ", StatePriorDistribution(cat_dp))
println("CAT WORLD POSTERIOR STATE PDF: ", StatePosteriorDistribution(cat_dp, first_state_space[1], 0.68))
println("STATE INDEX: ", POMDPs.state_index(cat_dp, test_state))

### TEST: ACTION
println("CAT WORLD ACTION SPACE: ", action_space)
println("ACTION INDEX: ", get_action_index(cat_dp, :listen_to_doors))

### TEST: OBSERVATION
println("CAT WORLD PRIOR OBSERVATION PDF: ", ObservationPriorDistribution(cat_dp))
println("CAT WORLD POSTERIOR STATE PDF: ", ObservationPosteriorDistribution(cat_dp, observation_space[1], 0.68))
# println("OBSERVATION INDEX: ", POMDPs.state_index(cat_dp, test_state))

### TEST: REWARD
println("REWARD: ", get_reward(cat_dp, first_state_space[1], action_space[1]))



############################################## QPMD

### SOLVER: QMDP
@requirements_info QMDPSolver() cat_dp

qmdp_policy = solve(QMDPSolver(max_iterations=50, tolerance=1e-3), cat_dp, verbose=true)
println("QMDP_POLICY: \n", qmdp_policy)
qmdp_belief_updater = updater(qmdp_policy)
println("QMDP_UPDATE_BELIEF: \n", qmdp_belief_updater)

### POLICY: QMDP
print("QMDP_POLICY_ALPHAS: \n", qmdp_policy.alphas, "\n\n")
qmdp_belief_test = DiscreteBelief(2); # initial uniform over two states
qmdp_action_test = action(qmdp_policy, qmdp_belief_test)
println("QMDP_TEST_BELIEF=", qmdp_belief_test, " ==> ACTION=", qmdp_action_test, "\n")


############################################## SIMULATE: QMDP

### SIMULATION: QMDP
# simulate{S,A,O,B}(simulator::Simulator, problem::POMDP{S,A,O}, policy::Policy{B}, updater::Updater{B}, initial_belief::B)
# simulate{S,A}(simulator::Simulator, problem::MDP{S,A}, policy::Policy, initial_state::S)
# http://juliapomdp.github.io/POMDPs.jl/latest/api/#POMDPs.simulate
qmdp_history_simulator = HistoryRecorder(max_steps=14)  # , rng=MersenneTwister(1)
# removed per Zachary: POMDPs.initial_state_distribution(cat_dp)
# debug here: /Users/pablo/.julia/v0.6/POMDPToolbox/src/simulators/history_recorder.jl
qmdp_simulated_history = simulate(qmdp_history_simulator, cat_dp, qmdp_policy, qmdp_belief_updater, StatePriorDistribution(cat_dp))
# println("Total reward: $(discounted_reward(qmdp_simulated_history)) \n")


# ############################################## SARSOP: all time favorite, only care about belief states reachable by optimal policy (up to 10k states)
#
# ### SOLVER: SARSOP
# # No requirements specified
# @requirements_info SARSOPSolver() TigerDp()
#
# # sarsop_policy = solve(SARSOPSolver(), cat_dp)   # SARSOP Discrete Bayesian Filter http://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/
# # println(sarsop_policy)
#
# # sarsop_history_recorder = HistoryRecorder(max_steps=14, rng=MersenneTwister(1))                # history recorder that keeps track of states, observations and beliefs
# # belief_updater = updater(sarsop_policy)
# # sarsop_simulated_history = simulate(sarsop_history_recorder, cat_dp, qmdp_policy, belief_updater, prior_distribution)
#
# # print("POLICY ALPHAS: \n", policy.alphas, "\n\n")
# # sarsop_belief_test = DiscreteBelief(2); # the initial prior
# # sarsop_action_test = action(sarsop_policy, sarsop_belief_test)
# # println("TEST: BELIEF=", sarsop_belief_test, " ==> ACTION=", sarsop_action_test, "\n")
#
# #println("Total reward: $(discounted_reward(hist)) \n")
#
#
# ############################################## SIMULATE

## SIMULATE
t=1
for (s, b, a, r, sp, o) in qmdp_simulated_history
     println("t=",t, "\t", "s=$s, b=$(b), a=$a, o=$o, r=$r", "\n")

     # println("t=",t, "\t", "s=$s, b=$(b.b), a=$a, o=$o, r=$r", "\n")
#     state = state_name[s]
#     println("\t(b,s) believe_where[L,R]=$(b.b) <==> actually_where=", s)
#     t+=1
# #    println("t=",t)
# #    println("\t(a)=", a)
# #    println("\t(o)=", o)
end



# #for (s, a, o, r) in stepthrough(cat_dp, policy, "s,a,o,r", max_steps=10)
# #    println("in state $s")
# #    println("took action $o")
# #    println("received observation $o and reward $r")
# #end
#
#
#
#
# #policy = RandomPolicy(cat_dp)
