function  [x,y,z] = get_3D(m1,m2,u1,v1,u2)

A = zeros(3, 3);
B = zeros(3, 1);

A(1,1) = u1 * m1(3,1) - m1(1,1);
A(1,2) = u1 * m1(3,2) - m1(1,2);
A(1,3) = u1 * m1(3,3) - m1(1,3);

A(2,1) = v1 * m1(3,1) - m1(2,1);
A(2,2) = v1 * m1(3,2) - m1(2,2);
A(2,3) = v1 * m1(3,3) - m1(2,3);

A(3,1) = u2 * m2(3,1) - m2(1,1);
A(3,2) = u2 * m2(3,2) - m2(1,2);
A(3,3) = u2 * m2(3,3) - m2(1,3);

B(1,1) = m1(1, 4) - u1 * m1(3,4);
B(2,1) = m1(2, 4) - v1 * m1(3,4);
B(3,1) = m2(1, 4) - u2 * m2(3,4);

points3D = A\B;
x=points3D(1);
y=points3D(2);
z=points3D(3);







