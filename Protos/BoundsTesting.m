function [loss,gradients] = modelLoss(net,X,T,X0,T0,U0)

% Make predictions with the initial conditions.
XT = cat(1,X,T);
U = forward(net,XT);

disp(size(X))
disp(size(T))

% Calculate derivatives with respect to X and T.
X = stripdims(X);
T = stripdims(T);
U = stripdims(U);

disp("---")
disp(size(X))
disp(size(T))
disp(size(XT))

Ux = dljacobian(U,X,1);
Ut = dljacobian(U,T,1);

% Calculate second-order derivatives with respect to X.
Uxx = dldivergence(Ux,X,1);

% Calculate mseF. Enforce Burger's equation.
f = Ut + U.*Ux - (0.01./pi).*Uxx;
mseF = mean(f.^2);

% Calculate mseU. Enforce initial and boundary conditions.
XT0 = cat(1,X0,T0);
U0Pred = forward(net,XT0);
mseU = l2loss(U0Pred,U0);

% Calculated loss to be minimized by combining errors.
loss = mseF + mseU;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end

numBoundaryConditionPoints = [25 25];

x0BC1 = -1*ones(1,numBoundaryConditionPoints(1));
x0BC2 = ones(1,numBoundaryConditionPoints(2));

t0BC1 = linspace(0,1,numBoundaryConditionPoints(1));
t0BC2 = linspace(0,1,numBoundaryConditionPoints(2));

u0BC1 = zeros(1,numBoundaryConditionPoints(1));
u0BC2 = zeros(1,numBoundaryConditionPoints(2));

numInitialConditionPoints  = 50;

x0IC = linspace(-1,1,numInitialConditionPoints);
t0IC = zeros(1,numInitialConditionPoints);
u0IC = -sin(pi*x0IC);

X0 = [x0IC x0BC1 x0BC2];
T0 = [t0IC t0BC1 t0BC2];
U0 = [u0IC u0BC1 u0BC2];

numInternalCollocationPoints = 10000;

points = rand(numInternalCollocationPoints,2);

dataX = 2*points(:,1)-1;
dataT = points(:,2);

numBlocks = 8;
fcOutputSize = 20;

fcBlock = [
    fullyConnectedLayer(fcOutputSize)
    tanhLayer];

layers = [
    featureInputLayer(2)
    repmat(fcBlock,[numBlocks 1])
    fullyConnectedLayer(1)];

net = dlnetwork(layers);

net = dlupdate(@double,net);

solverState = lbfgsState;
maxIterations = 1500;
gradientTolerance = 1e-5;
stepTolerance = 1e-5;

X = dlarray(dataX,"BC");
T = dlarray(dataT,"BC");
X0 = dlarray(X0,"CB");
T0 = dlarray(T0,"CB");
U0 = dlarray(U0,"CB");

%accfun = dlaccelerate(@modelLoss);

%lossFcn = @(net) dlfeval(accfun,net,X,T,X0,T0,U0);

modelLoss(net,X,T,X0,T0,U0)