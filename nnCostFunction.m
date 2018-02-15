function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (	input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));





for i =1:size(y)
ym(i,y(i)) = 1;
end;
Xm = [ones(m,1) X];
fp = Xm*Theta1';
sfp = sigmoid(fp);
sfp = [ones(m,1) sfp];	

sp = sfp*Theta2';
 ssp = sigmoid(sp);
 lssp = log(ssp);
frstpart = lssp.*ym;
prt2lssp = log(1-ssp);
scndpart = prt2lssp.*(1-ym);

final =   (frstpart + scndpart);
final_sum = 	-1*sum(final(:))/m;




tempt1 = (Theta1.*Theta1);

tempt1subt = sum((Theta1(:,1).*Theta1(:,1))');
tempt2subt = sum((Theta2(:,1).*Theta2(:,1))');

tempt2 = (Theta2.*Theta2);
temptheta1 = (sum(tempt1(:))  - tempt1subt )*(lambda/(2*m));
temptheta2 = (sum(tempt2(:))  - tempt2subt)*(lambda/(2*m));

J = temptheta1+temptheta2+final_sum ;





del3 = ssp - ym ;

theta2temp = Theta2;
%theta2temp(:,1 )  = [];
tempsfp = sfp;

del2 = (del3*theta2temp).*(tempsfp.*(1-tempsfp) );
del2(:,1 )  = [];

DEL2 = (del3'*sfp)/m;
DEL1 = (del2'*Xm)/m;

Theta1_grad = DEL1  ;
Theta2_grad = DEL2  ;

Theta1_grad(:,2:end) = DEL1(:,2:end) + ((Theta1(:,2:end))*lambda)/m;
Theta2_grad(:,2:end) = DEL2(:,2:end) + ((Theta2(:,2:end))*lambda)/m;





















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
