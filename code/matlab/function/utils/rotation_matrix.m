function R = rotation_matrix(theta)

    % 2D
    % Rz = zeros(3);
    % Rz(3,3) = 1;
    % Rz(1:2,1:2) = [cos(theta) -sin(theta); sin(theta) cos(theta)];   


    Rx = [1           0           0;
          0           cos(theta)  -sin(theta); 
          0           sin(theta)  cos(theta)];

    Ry = [cos(theta)  0           sin(theta);
          0           1           0; 
          -sin(theta) 0           cos(theta)];

    Rz = [cos(theta)  -sin(theta) 0;
          sin(theta)  cos(theta)  0; 
          0           0           1];

    R = Rz*Ry*Rx;

%     R = Rz;
end