import os
import sys
import crocoddyl
import pinocchio
import numpy as np
import example_robot_data

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ
WITHDISPLAY = True
robot = example_robot_data.load('tiago_no_hand_head_fixed')
#robot = example_robot_data.loadTiago(True,'wsg')
robot_model = robot.model


DT = 1e-3
T= 250
target = np.array([3.8, 0.1, 1.1]) 

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
qnew = [*q0[0:7] , *[0.0,0.0,0.0]]
# calculate angle of the wheel
qw1 = np.arctan2(q0[8],q0[9])
qw2 = np.arctan2(q0[10],q0[11])


 
x0 = np.concatenate([q0, pinocchio.utils.zero(state.nv)])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverFDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

cameraTF = [3., 2.68, 1.54, 0.2, 0.62, 0.72, 0.22]

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



 # Visualizing the solution in gepetto-viewer
display.displayFromSolver(ddp)

robot_data = robot_model.createData()
xT = ddp.xs[-1]
pinocchio.forwardKinematics(robot_model, robot_data, xT[:state.nq])
pinocchio.updateFramePlacements(robot_model, robot_data)
print('Finally reached = ', robot_data.oMf[robot_model.getFrameId("wrist_tool_joint")].translation.T)


# Plotting the solution and the DDP convergence
#if WITHPLOT:
#    log = ddp.getCallbacks()[0]
#    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
#    crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)
#
## Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(robot)
    display.displayFromSolver(ddp)