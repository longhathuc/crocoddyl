import os
import sys
import crocoddyl
import pinocchio
import numpy as np
import example_robot_data

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

robot = example_robot_data.load('tiago')
#robot = example_robot_data.loadTiago(True,'wsg')
robot_model = robot.model
q0 = robot_model.referenceConfigurations["tuck_arm"]
# x0 = np.concatenate([q0, np.zeros(robot_model.nv)])
# robot.q0 = q0
DT = 1e-3
T= 250
target = np.array([0.8, 0.1, 1.1]) 
display = crocoddyl.MeshcatDisplay(robot, 4, 4, False)

print("referenceConfigurations:", q0)
print("robot.q0 size:", robot.q0.shape[0])
print(robot.q0)

# Create the cost functions
Mref = crocoddyl.FrameTranslation(robot_model.getFrameId("wrist_tool_joint"), target)
state = crocoddyl.StateMultibody(robot.model)
goalTrackingCost = crocoddyl.CostModelFrameTranslation(state, Mref)
xRegCost = crocoddyl.CostModelState(state)
uRegCost = crocoddyl.CostModelControl(state)

# Create cost model per each action model
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e2)
runningCostModel.addCost("stateReg", xRegCost, 1e-4)
runningCostModel.addCost("ctrlReg", uRegCost, 1e-7)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e5)
terminalCostModel.addCost("stateReg", xRegCost, 1e-4)
terminalCostModel.addCost("ctrlReg", uRegCost, 1e-7)

# Create the actuation model
actuationModel = crocoddyl.ActuationModelFull(state)

# Create the action model
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, runningCostModel), DT)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, terminalCostModel))


# Create the problem
q0 = robot_model.referenceConfigurations["tuck_arm"]
x0 = np.concatenate([q0, pinocchio.utils.zero(state.nv)])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverDDP(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(robot, 4, 4, cameraTF)
    ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(robot, 4, 4, cameraTF)
    ddp.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    ddp.setCallbacks([
        crocoddyl.CallbackLogger(),
        crocoddyl.CallbackVerbose(),
    ])
else:
    ddp.setCallbacks([crocoddyl.CallbackVerbose()])


# Solving it with the DDP algorithm
ddp.solve()


# Plotting the solution and the DDP convergence
if WITHPLOT:
    log = ddp.getCallbacks()[0]
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(robot, 4, 4, cameraTF)
    display.displayFromSolver(ddp)


# # Visualizing the solution in gepetto-viewer
# display.displayFromSolver(ddp)

# robot_data = robot_model.createData()
# xT = ddp.xs[-1]
# pinocchio.forwardKinematics(robot_model, robot_data, xT[:state.nq])
# pinocchio.updateFramePlacements(robot_model, robot_data)
# print('Finally reached = ', robot_data.oMf[robot_model.getFrameId("wrist_tool_joint")].translation.T)