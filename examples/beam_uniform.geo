// This code was created by pygmsh vunknown.
p0 = newp;
Point(p0) = {0.0, 0.0, 0.0, 1.0};
p3 = newp;
Point(p3) = {100.0, 0.0, 0.0, 1.0};
p4 = newp;
Point(p4) = {100.0, 16.0, 0.0, 0.08};
p5 = newp;
Point(p5) = {100.0, 24.0, 0.0, 0.08};
p6 = newp;
Point(p6) = {100.0, 40.0, 0.0, 1.0};
p9 = newp;
Point(p9) = {0.0, 40.0, 0.0, 1.0};//+
//+
Line(1) = {6, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Line(5) = {4, 5};
//+
Line(6) = {5, 6};
//+
Curve Loop(1) = {6, 1, 2, 3, 4, 5};
//+
Plane Surface(1) = {1};
//+
Transfinite Surface {1} = {6, 5, 2, 1};
//+
Transfinite Curve {6, 2} = 26 Using Progression 1;
//+
Transfinite Curve {1} = 11 Using Progression 1;
//+
Transfinite Curve {5, 3} = 5 Using Progression 1;
//+
Transfinite Curve {4} = 3 Using Progression 1;
//+
Physical Curve(3) = {1};
//+
Physical Curve(4) = {4};
Physical Surface(1) = {1};
