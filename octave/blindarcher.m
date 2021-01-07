## Copyright (C) 2020 juanlu
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} patternsearch (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: juanlu <juanlu@verne>
## Created: 2020-09-14

% --------------------  Pattern Search --------------------------------  
% x0 a column vector of N dimensions (typically randomly initialized)
% alpha0 the initial step size
% objectivefunction can be:
%                           - sphere
%                           - ellipsoid
%                           - rotatedEllipsoid
%                           - rosenbrock
%                           - bentcigar
%                           - bentcigarshiftedrotated 
% basis can be:
%                           - simplex
%                           - standard/cartesian (default)
% order can be:
%                           - same (always follow the same order in trying points)
%                           - random (points are selected in a random permutation)
%                           - successful (start moving in the last successful direction)
% tauplus is the ratio for augmenting the step when successful
% tauminus is the ratio for decreasing the step after failure

function [retval] = blindarcher(x0,alpha0,objectivefunction,basis,order,tauplus,tauminus)
  
  % --------------------  Initialization --------------------------------  
  % User defined input parameters (need to be edited)
  clc
  warning off
  global strfitnessfct;  % name of objective/fitness function
  global N;               % number of objective variables/problem dimension
  global alphak;
  global xk;
  global k;                         %Iteration number
  global numberevaluations;
  global ksuccessful;
  global B;
  global fxk;
  global Succ; % Matrix to keep successful evaluations
  global Unsucc; % Matrix to keep successful evaluations
  global tabu; % Not to evaluate the last successful move
  global successfulDirection;
  global RmatrixFunction;
  global thetak;
  
  % Testing rotations
  %Basis = eye(3);
  %pos = 2;
  %theta = pi/4;
  %rotations(Basis,pos,theta);
  %return;
  
  RmatrixFunction = zeros (3, 0);
  
  if nargin == 0
    x0 = (10.*rand(2,1))-5; alpha0=0.2; objectivefunction='rotatedEllipsoid';
    basis = 'standard'; order = 'successful'; tauplus = 1; tauminus = 0.5;
  elseif nargin < 2
    alpha0=0.2; objectivefunction='sphere';
    basis = 'standard'; order = 'successful'; tauplus = 1; tauminus = 0.5;
  elseif nargin < 3
    objectivefunction='sphere';
    basis = 'standard'; order = 'successful'; tauplus = 1; tauminus = 0.5;
  elseif nargin < 4
    basis = 'standard'; order = 'successful'; tauplus = 1; tauminus = 0.5;
  elseif nargin < 5
    order = 'successful'; tauplus = 1; tauminus = 0.5;
  elseif nargin < 6
    tauplus = 1; tauminus = 0.5;
  elseif nargin < 7
    tauminus = 0.5;
  endif
  
  
  strfitnessfct = objectivefunction;
  N = size(x0,1);
  alphak = alpha0;
  xk = x0;
  k = 0;
  ksuccessful = 0;
  if  strcmp(basis,'simplex')
    B = simplex(N+1);
  else %standard/cartesian basis
    B = horzcat(eye(N),-eye(N));
  endif
  
  fxk = feval(strfitnessfct, xk);
  numberevaluations = 1;
  tabu = xk;
  Succ = vertcat(xk,fxk);
  Unsucc =  vertcat(xk,fxk);
  
  %printf("Iter %d fx %f Vector %s \n",k,fxk,mat2str(transpose(xk)));
  %printf("Iter %d fx %f\n",k,fxk);
  
  % -- Prepearing for pollstep
  ord = order;
  taup = tauplus;
  taum = tauminus;
  
  % In the case that we want to prioritaze the order
  % of evaluation in the mesh starting from the last 
  % successfully evaluated direction we need to keep 
  % this information.
  % A priori we do not have any knowledge about so we
  % initialize to whatever direction of the basis.
  successfulDirection = 1; 

  
  % Rotation
  thetak = pi/8;
  
  % --------------------  End Initialization ---------------------------- 
  
  
  % --------------------  Main loop -------------------------------------
  %printf("Time,iterations,evaluations,fitness \n");
  start = time();
  leadingVectorBasis();
  while (k < 10000) && (fxk > 0.01)
    if ( pollstep(ord,taup,taum) == 0)
      %leadingVectorBasis();
    endif
  endwhile
  stop = time();
  printf("%f,%d,%d,%f\n",(stop-start),k,numberevaluations,fxk);
  % --------------------  End main loop ---------------------------------
  
  % Plotting for 2D functions
  if (N == 2)
    plot3D();
  endif
  
  retval = numberevaluations;
  clear strfitnessfct N alphak xk k numberevaluations ksuccessful B fxk Succ Unsucc successfulDirection tabu RmatrixFunction;  
endfunction

% --------------------  Poll step --------------------------------  
function leadingVectorBasis()
  global B;
  global xk;
  global alphak;
  global strfitnessfct;
  global successfulDirection;
  global numberevaluations;
  global fxk;
  global Succ;
  global thetak;
  
  leadingDirection = 0;
  xk1 = xk;
  fxk1 = realmax;
  
  
  for i = 1:columns(B)
    xpp = xk + alphak .* B(:,i);
    newfval = feval(strfitnessfct, xpp);
    numberevaluations ++;
    
    Succ = horzcat(Succ,vertcat(xpp,newfval));
    
    if (newfval < fxk1)
        %disp("-------");
        leadingDirection = i;
        xk1 = xpp;
        fxk1 = newfval;
        %disp("-------");
    endif
  endfor
  
  successfulDirection = leadingDirection;
  
%  if ( fBest < fxk )
%    xk = best;
%    fxk = fBest;
%  endif
  
  [x,fx] = rotationstep(leadingDirection,xk1,fxk1,pi/8);
  Succ = horzcat(Succ,vertcat(x,fx));
  [x,fx] = rotationstep(leadingDirection,x,fx,pi/16);
  Succ = horzcat(Succ,vertcat(x,fx));
  [x,fx] = rotationstep(leadingDirection,x,fx,pi/32);
  Succ = horzcat(Succ,vertcat(x,fx));
  
 
endfunction

% --------------------  Poll step --------------------------------  
function ksuccessful = pollstep(order,tauplus,tauminus)
  global strfitnessfct;  % name of objective/fitness function
  global N;               % number of objective variables/problem dimension
  global alphak;
  global xk;
  global k;
  global numberevaluations;
  global ksuccessful;
  global B;
  global fxk;
  global Succ; % Matrix to keep successful evaluations
  global Unsucc; % Matrix to keep successful evaluations
  global tabu;
  global successfulDirection;
  global thetak;
  
  
  % Generating order of evaluation
  if strcmp(order,'successful')
    orderOfEvalution = horzcat(successfulDirection:columns(B),1:(successfulDirection-1));
  elseif strcmp(order,'random')
    orderOfEvalution = randperm(columns(B));
  else % Default is 'same' which stands for always the same order of evaluation
    orderOfEvalution = 1:columns(B);
  endif

  % Polling step procedure
  ksuccessful = 0;
  directionk1 = 0;
  xk1 = xk;
  fxk1 = realmax;
  
  for i = orderOfEvalution
  %for i = successfulDirection
    xpp = xk + alphak .* B(:,i);
    
    if ! isequal(xpp, tabu)
      newfval = feval(strfitnessfct, xpp);
      numberevaluations ++;
      
      if (newfval < fxk1)
        directionk1 = i;
        xk1 = xpp;
        fxk1 = newfval;
      endif
      
      if ( newfval < fxk )
        ksuccessful = 1;
        successfulDirection = i;
        Succ = horzcat(Succ,vertcat(xpp,newfval));
        break;
      else
        Unsucc = horzcat(Unsucc,vertcat(xpp,newfval));
      endif 
    endif
  end
  
  % Mesh procedure
  if ksuccessful
    tabu = xk;
    xk = xpp;
    fxk = newfval;
    alphak = alphak * tauplus;
    thetak = pi/8;
    if (successfulDirection != orderOfEvalution(1))
      %[x, fx] = rotationstep(directionk1,xk1,fxk1,thetak);
    endif

   else
    alphak = alphak * tauminus;
    [x, fx] = rotationstep(directionk1,xk1,fxk1,thetak);
    thetak = thetak/2;
  endif
  
  % Increase iterations
  k = k + 1;
  
  %printf("Iter %d fx %f Vector %s \n",k,fxk,mat2str(transpose(xk)));
  %printf("Iter %d fx %f\n",k,fxk);
  
endfunction

% --------------------  Rotation step --------------------------------  
function [bestx, fxbest] = rotationstep(directionk1,xk1,fxk1,theta)
  global B;
  global xk;
  global fxk;
  global alphak;
  global numberevaluations;
  global strfitnessfct;
  
  % Basis = first half of B
  dim = size(B,1);
  Basis = B(:,1:dim);
  pos = directionk1;
  if directionk1 > dim
    pos = directionk1 - dim;
  end
  
  % Obtain all the rotation matrices
  RotationMatrices = rotations(Basis,pos,theta);
  
  % Compute the best rotation matrix
  bestBasis = B;
  bestx = xk1;
  fxbest = fxk1;
  for i = 1: 2*(dim-1)
    auxBasis = RotationMatrices(:,:,i) * B;
    x = xk + alphak .* auxBasis(:,directionk1);
    newfval = feval(strfitnessfct, x);
    numberevaluations ++;
    
    if ( newfval < fxbest )
      bestBasis = auxBasis;
      bestx = x;
      fxbest = newfval;
    end
  end
  
  B = bestBasis;
  
  %if (f_best < fxk)
  %  xk = f_x;
  %  fxk = f_best;
  %endif
  
endfunction

% --------------------  Rotations --------------------------------  
% return 2 x (n-1) rotation matrices of the Basis around the vector B(:,pos)
% with angle theta

function [RotationMatrices] = rotations (Basis,pos,theta)
  dim = size(Basis,1);
  % Building all the rotated Basis V
  
  % Prepearing basis v to span n-2 subspace
  v = zeros(dim,dim-2,dim-1);
  
  % Prepearint basis without the leading vector
  Bp = horzcat(Basis(:,1:(pos-1)),Basis(:,(pos+1):end));

  % Indices of combinations without repetitions
  index = nchoosek(1:(dim-1),dim-2);
  dimMinus1 = size(index,1);
  dimMinus2 = size(index,2);
  
  % Creating the basis v from the basis Bp
  for i = 1: dimMinus1
    for j = 1: dimMinus2
      %index(i,j)
      v(:,j,i) = Bp(:,index(i,j));
    endfor
  endfor
  
  
  % Creating 2 x n-1 rotation matrices
  RotationMatrices = zeros (dim,dim,2*dimMinus1);
  for i = 1:2:(2*dimMinus1)
  %for i = 1:dimMinus1
    %RotationMatrices(:,:,i) =  rotmnd(v(:,:,i),theta);
    RotationMatrices(:,:,i) =  rotmnd(v(:,:,fix((i+1)/2)),theta);
    RotationMatrices(:,:,i+1) =  rotmnd(v(:,:,fix((i+1)/2)),-theta);
  endfor
  
  if dim == 2
    v = [0;0];
    RotationMatrices(:,:,1) =  rotmnd(v,theta);
    RotationMatrices(:,:,2) =  rotmnd(v,-theta);
  endif
endfunction



% ---------------------------------------------------------------  
% --------------------  Functions  ------------------------------  
% ---------------------------------------------------------------  

% --------------------  Rosenbrock  -----------------------------  
function f=rosenbrock(x)
    if size(x,1) < 2 error('dimension must be greater one'); end
    f = 100*sum((x(1:end-1).^2 - x(2:end)).^2) + sum((x(1:end-1)-1).^2);
end


% --------------------  Ellipsoid  ---------------------------------  
function f=ellipsoid(x)
  dim = size(x,1);
  if dim < 1 error('dimension must be greater than zero'); end
  f = 0;
  for i = 1:dim
    f = f + (50*(i^2*x(i))^2);
  endfor
end

% --------------------  Ellipsoid Rotated pi/7 ---------------------  
% --- Rotation is done in all the planes
function f=rotatedEllipsoid(x)
  global RmatrixFunction;
  dim = size(x,1);
  if dim < 1 error('dimension must be greater than zero'); end

  % Rotating x  
  if isempty(RmatrixFunction)
    nrot = nchoosek (dim, 2); % Number of rotations combinations dim over 2
    thetas = pi/7 .* ones(nrot);
    RmatrixFunction = rotateAngles(dim,thetas);
  endif
  xrot = RmatrixFunction * x;
  
  f = 0;
  for i = 1:dim
    f = f + (50*(i^2*xrot(i))^2);
  endfor
end



% --------------------  Sphere  ---------------------------------  
function f=sphere(x)
  if size(x,1) < 1 error('dimension must be greater than zero'); end
  f = sum((x(1:end).^2));
end

% --------------------  Bentcigar  -------------------------------  
function f=bentcigar(x)
  if size(x,1) < 2 error('dimension must be greater than one'); end
  f = x(1).^2 + (10^6) * sum((x(2:end).^2));
end

% --------------------  Bentcigar shifted and rotated  -----------  
function f=bentcigarshiftedrotated(x)
  global RmatrixFunction;
  dim = size(x,1);
  if dim < 2 error('dimension must be greater than one'); end

  % Rotating x  
  if isempty(RmatrixFunction)
    nrot = nchoosek (dim, 2); % Number of rotations cominations dim over 2
    thetas = pi/7 .* ones(nrot);
    RmatrixFunction = rotateAngles(dim,thetas);
  endif
  xrot = RmatrixFunction * x;

  % Shifting the rotated x
  xshift = xrot .+ 10;
  
  f = xshift(1).^2 + (10^6) * sum((xshift(2:end).^2));
end


% ---------------------------------------------------------------  
% --------------------  Auxiliary functions  --------------------  
% ---------------------------------------------------------------  

% --------------------  Simplex --------------------------------  
function V=simplex(N)
  % create regular simplex with N vertecies and N-1 dimentional space
  % V - vertecies coordinates
  % size(V)=[N-1 N]
  % V(dimention_number,vertex_number)
  % algorithm: http://en.wikipedia.org/wiki/Simplex#Cartesian_coordinates_for_regular_n-dimensional_simplex_in_Rn
  % The coordinates of the vertices of a regular n-dimensional simplex can be obtained from these two properties,
  %1)For a regular simplex, the distances of its vertices to its center are equal.
  %2) The angle subtended by any two vertices of an n-dimensional simplex
  % through its center is arccos(-1/(N-1))
  V=zeros(N-1,N);


  for nc=1:N-1
      V(nc,nc)=sqrt(1-sum([V(1:nc-1,nc); V(nc+1:end,nc)].^2)); % because |V(nc,:)|=1 (sum(V(nc,:).^2)=1)
      % here all V(nc,:) known, it has 1:nc - non-zeros and nc+1:end -zeros
      % now need to find V(nc,nc+1),V(nc,nc+2),...,V(nc,end) unding arccos(-1/(N-1)):
      % u(1)*v(1)+u(2)*v(2)+...u(nc-1)*v(nc-1)+u(nc)*v(nc)=-1/(N-1)
      % v(nc)=-(u(1)*v(1)+u(2)*v(2)+...+u(nc-1)*v(nc-1)+1/N)/u(nc)
      % u is V(:,nc)
      % v can be any of V(:,nc) V(:,nc+1) ... V(:,end)
      for nc1=nc+1:N
        V(nc,nc1)=-(sum(V(1:nc-1,nc).*V(1:nc-1,nc1))+1/(N-1))/V(nc,nc);
      endfor
  endfor
end
% to check:
% see V'*V - symmetrical with 1 at diagonal and -1/(N-1) in rest
% mean(V,2) vector close to zero vector

% ---------------------------------------------------------------  
% --------------------  Rotations  ------------------------------  
% --------------------------------------------------------------- 

% Implementation of the Aguilera-Perez Algorithm.
% Aguilera, Antonio, and Ricardo Pérez-Aguila. "General n-dimensional rotations." (2004).
% Found here : https://stackoverflow.com/questions/50337642/how-to-calculate-a-rotation-matrix-in-n-dimensions-given-the-point-to-rotate-an
% https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.4.8662
% Aguilera, Antonio; Peréz-Aguila, Ricardo; General n-dimensional rotations
% WSCG '2004: Short Communications: the 12-th International Conference in Central Europe on Computer Graphics, Visualization and Computer Vision 2004, 2.-6. February 2004 Plzeň, p. 1-8.
function M = rotmnd(v,theta)
    n = size(v,1);
    M = eye(n);
    for c = 1:(n-2)
        for r = n:-1:(c+1)
            t = atan2(v(r,c),v(r-1,c));
            R = eye(n);
            R([r r-1],[r r-1]) = [cos(t) -sin(t); sin(t) cos(t)];
            v = R*v;
            M = R*M;
        endfor
    endfor
    R = eye(n);
    R([n-1 n],[n-1 n]) = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    M = M\R*M;
end

% This function returns all the possible rotations planes
% for a given dimension - combinatorics of dim over 2
% If you want to select the n-th basis do: v(:,:,n)
function v = rotationBasis(dim)
  if dim == 2
    v = [0;0];
    return;
  endif
  
  nrot = nchoosek (dim, 2); % Number of posible rotations planes
  v = zeros(dim,dim-2,nrot);
  diag = eye(dim-2);

  rot = 1;
  for i = 1 : dim - 1
    for j = i + 1 : dim
      %Rotation 1,2,3,....
      count = 1;
      for h = 1:dim
        if h != i && h != j
          v(h,:,rot) = diag(count,:);
          count++;
        endif
      endfor
      rot++;
    endfor
  endfor
end

% Rotate in a dim space - 2D, 3D,... - all the angles in thethas
% Thethas is a vector for all the possible rotation planes in dim space
% The length of thetas is the combinatorics of dim over 2
function R = rotateAngles(dim,thetas)
  v = rotationBasis(dim);
  R = eye(dim);
  nrot = nchoosek (dim, 2);
  for i = 1: nrot
    R = R * rotmnd(v(:,:,i),thetas(i));
  endfor
end
% --------------------  Plotting  -------------------------------  
function plot3D()
  global strfitnessfct;
  global xk;
  global Succ;
  global Unsucc;
  
  x1min=-5;
  x1max=5;
  x2min=-5;
  x2max=5;
  R=150; % steps resolution
  x1=x1min:(x1max-x1min)/R:x1max;
  x2=x2min:(x2max-x2min)/R:x2max;
  
  for j=1:length(x1)
    for i=1:length(x2)
        x = vertcat (x1(j),x2(i));
        f(i) = feval(strfitnessfct, x);
        printf ("",x1(j),x2(i),f(i));
        fit(i,j) = f(i);
    end
    f_tot(j,:)=f;
  end
  

  
  figure(1)
  plot3(Succ(2,1:end),Succ(1,1:end),Succ(3,1:end),'color','red','linewidth',5);
  hold on;
  meshc(x1,x2,f_tot);colorbar;set(gca,'FontSize',12);
  xlabel('x_2','FontName','Times','FontSize',20,'FontAngle','italic');
  set(get(gca,'xlabel'),'rotation',25,'VerticalAlignment','bottom');
  ylabel('x_1','FontName','Times','FontSize',20,'FontAngle','italic');
  set(get(gca,'ylabel'),'rotation',-25,'VerticalAlignment','bottom');
  zlabel('f(X)','FontName','Times','FontSize',20,'FontAngle','italic');
  title(strfitnessfct,'FontName','Times','FontSize',24,'FontWeight','bold');
  hold off;
  
  figure(2)
  hold on;
  contour(x1,x2,f_tot);
  %plot(xk(1),xk(2),"s");
  plot(Succ(2,1:end),Succ(1,1:end),"s");
  %plot(Unsucc(1,1:end),Unsucc(2,1:end),"*");
  hold off;
  
end
