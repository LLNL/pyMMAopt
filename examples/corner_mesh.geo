Point(1) = {0., 0., 0., 0.01};
Point(2) = {10., 0., 0., 0.5};
Point(3) = {10., 10., 0., 0.5};
Point(4) = {0., 10., 0., 0.5};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {3, 4, 1, 2};
//+
Plane Surface(1) = {1};
