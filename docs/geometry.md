# Geometry notes

All distance figures in this project come from the pinhole camera model. This page derives each
formula the code uses, names the assumptions, and states the error each assumption introduces.

## 1. Pinhole model

A point $(X, Y, Z)$ in the camera frame (X right, Y down, Z forward, metres) lands on pixel

$$u = f_x \frac{X}{Z} + c_x, \qquad v = f_y \frac{Y}{Z} + c_y$$

where $f_x, f_y$ are the focal lengths in pixels and $(c_x, c_y)$ is the principal point.
`CameraIntrinsics` in `geometry.py` holds these four numbers together with the frame size they
belong to. When a frame is resized, every one of them scales by the same ratio, which is what
`CameraIntrinsics.scaled` does. When only the field of view is known,

$$f_x = \frac{W / 2}{\tan(\mathrm{HFOV} / 2)}$$

(`CameraIntrinsics.from_hfov`). The nuImages front camera is $f \approx 1266$ px at 1600 x 900,
a horizontal field of view of about 64.5 degrees.

## 2. Distance from the object's known height (default method)

A cone of real height $H$ metres at depth $Z$ covers $h = f_y H / Z$ pixels, so

$$Z = \frac{f_y \, H}{h}.$$

This is `distance_from_height`. It needs no camera height or pitch and works for objects that do
not touch the ground, such as a stop sign on a pole. The assumed heights live in `classes.py`:

| Class | Height used | Basis |
|---|---|---|
| cone | 0.70 m | the common 28 inch (0.71 m) road cone |
| barrier | 1.00 m | temporary concrete, plastic or metal road barriers, typically 0.8 to 1.1 m |
| stop_sign | 0.75 m | the 30 inch (0.76 m) octagon used on most roads |

Error model: the relative depth error equals the relative height error plus the relative box
height error. A cone that is really 0.45 m but assumed 0.70 m reads 55 percent too far; a box
that is 2 px too tall on a 20 px object reads 10 percent too near. Small far-away objects are
therefore the least reliable, which is why the distance is rounded to 0.1 m and the detection
confidence is kept beside it.

## 3. Distance from the ground contact point (optional method)

For objects standing on flat ground, the bottom edge of the box is at depth $Z$ and height
$-h_c$ below the camera (the camera sits $h_c$ metres above the ground), so for a level camera

$$v_{\text{bottom}} - c_y = f_y \frac{h_c}{Z} \quad\Rightarrow\quad Z = \frac{f_y \, h_c}{v_{\text{bottom}} - c_y}.$$

With a camera pitched down by $\theta$, the ray through the bottom pixel is depressed by
$\alpha = \arctan\big((v_{\text{bottom}} - c_y) / f_y\big) + \theta$ and meets the ground at
$Z = h_c / \tan\alpha$. A bottom pixel at or above the horizon ($\alpha \le 0$) gives no
distance. This is `distance_from_ground`. It does not need object sizes but needs $h_c$, the
pitch, flat ground and the box bottom to be the true contact point, which occlusion breaks.

## 4. Lateral offset and bearing

Once $Z$ is known, the horizontal pixel offset gives the sideways position and bearing:

$$X = \frac{(u - c_x)\, Z}{f_x}, \qquad \beta = \arctan\frac{u - c_x}{f_x}.$$

These are `lateral_offset` and `bearing_deg`, and they are what the bird's-eye view plots.

## 5. Epipolar geometry and the disparity to depth relation

Two cameras with intrinsics $K_l, K_r$ related by a rotation $R$ and translation $t$ (right
camera expressed in the left camera frame) satisfy, for every pair of matching pixels,

$$x_r^\top F x_l = 0, \qquad F = K_r^{-\top} [t]_\times R \, K_l^{-1}.$$

$F$ is the fundamental matrix, $[t]_\times$ the skew-symmetric cross-product matrix. The line
$l_r = F x_l$ is the epipolar line in the right image: the match of $x_l$ must lie on it, which
turns a 2D search into a 1D search. `fundamental_matrix` and `epipolar_line` implement this and
the tests check that a horizontal rig ($R = I$, $t = (-B, 0, 0)$) gives horizontal epipolar
lines.

For a rectified pair (parallel optical axes, same focal length $f$, baseline $B$), a point at
depth $Z$ and lateral position $X$ projects to

$$u_l = f \frac{X}{Z} + c_x, \qquad u_r = f \frac{X - B}{Z} + c_x.$$

Subtracting gives the disparity

$$d = u_l - u_r = \frac{f B}{Z} \quad\Rightarrow\quad Z = \frac{f B}{d}.$$

This is `depth_from_disparity`. The depth resolution degrades quadratically with distance:
differentiating, $|\Delta Z| = \frac{Z^2}{f B} |\Delta d|$, so a one pixel disparity error at
10 m on a $f = 700$ px, $B = 0.12$ m rig is a 1.2 m depth error, while at 2 m it is 5 cm.
`disparity_map` wraps OpenCV's semi-global block matcher and `depth_map` applies the relation.

## 6. Homography to a bird's-eye view

Ground points have a fixed height below the camera, so the ground plane and the image are
related by a homography. Put the world origin on the ground directly below the camera, X right,
Y down, Z forward. A ground point is $(X, 0, Z)$ and the camera is at $(0, -h_c, 0)$, so in a
level camera frame the point is $(X, h_c, Z)$. Pitching the camera down by $\theta$ rotates
this by $R_x(\theta)$, and the pixel is

$$\lambda \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = K R_x(\theta) \begin{pmatrix} X \\ h_c \\ Z \end{pmatrix}
= K \,[\, r_1 \mid r_3 \mid h_c r_2 \,] \begin{pmatrix} X \\ Z \\ 1 \end{pmatrix} = H \begin{pmatrix} X \\ Z \\ 1 \end{pmatrix}$$

with $r_i$ the columns of $R_x(\theta)$. `ground_to_image_homography` builds $H$ and
`bev_warp` composes it with the metric grid of the output image (one pixel per
$1 / \text{px\_per\_m}$ metres, forward up) and hands the product to `cv2.warpPerspective`.
Anything above the ground plane is smeared along its viewing ray in the warp, which is the usual
look of a bird's-eye view, so detections are drawn on the warp from their estimated $(X, Z)$
instead of being warped as pixels. `bev_from_points` is the four-point variant for cameras with
no calibration, where four known ground marks pin the homography directly.

## 7. Optical flow for tracking

`flow_points` runs pyramidal Lucas-Kanade on the centroid of every track, which gives where the
object moved between frames without running the detector. `FlowTracker` moves every track with
the flow, matches new detections to the moved positions by label and distance, and keeps a track
alive for a few frames when the detector misses it. Running the detector every $n$th frame and
flow in between is a common way to meet a frame-rate budget on a small CPU.
