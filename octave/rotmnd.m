% Implementation of the Aguilera-Perez Algorithm.
% Aguilera, Antonio, and Ricardo Pérez-Aguila. "General n-dimensional rotations." (2004).
% Found here : https://stackoverflow.com/questions/50337642/how-to-calculate-a-rotation-matrix-in-n-dimensions-given-the-point-to-rotate-an
% https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.4.8662
%
% Aguilera, Antonio; Peréz-Aguila, Ricardo; General n-dimensional rotations
% WSCG '2004: Short Communications: the 12-th International Conference in Central Europe on Computer Graphics, Visualization and Computer Vision 2004, 2.-6. February 2004 Plzeň, p. 1-8.
%
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
        end
    end
    R = eye(n);
    R([n-1 n],[n-1 n]) = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    M = M\R*M;
end