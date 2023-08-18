import torch

def ellipse_renderer(P, pixel_aug, pixel_norm_sq, rho=0.05, img_size=256, tau=100):
    
    #### Compute the points on the knot and their squared norms
    P = P.unsqueeze(1)                                                                              # Ns x 1 x 3
    P_norm_sq = P.square().sum(-1)                                                                  # Ns x 1
    
    #### Compute the ellipse occupancies
    dot1 = (pixel_aug * P).sum(-1).square().divide(P_norm_sq)                                       # Ns x M
    dot2 = (pixel_norm_sq).multiply(P_norm_sq.divide(P_norm_sq + rho**2))                           # Ns x M
    EP = torch.sigmoid(tau*(dot1 - dot2))                                                           # Ns x M
    
    #### Compte the rendering of the ellipse
    I_rendered = EP.max(0)[0].view(img_size,img_size)                                               # Ni x Ni
    return I_rendered

def capsule_renderer(P, pixels, rho=0.05, img_size=256, tau_ellipse=1000, tau_quadrilateral=100, fz=2, 
                        temp1_max=1e5, temp2_max=1e5, temp3_max=1e5, eps_z_pos=1e-2, eps_render=1e-8,
                        discriminant_sq_min=1e-4, alpha_max=1e5, beta_max=1e5, gamma_max=1e5):
    
    #### The x and y coordinates of the pixels on the image plane
    x = pixels.squeeze(0)[:, 0]
    y = pixels.squeeze(0)[:, 1]
    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)

    #### Compute the points on the knot and the constant term 'k^2' in the equation of the ellipse
    P = P.unsqueeze(1)
    P_norm_sq = P.square().sum(-1)
    k_sq = (P_norm_sq.square()).divide(P_norm_sq + rho**2).squeeze(-1)               # Ns x 1                               
    l = P.squeeze(-2)[:, 0]                                                                         # Ns 
    m = P.squeeze(-2)[:, 1]                                                                         # Ns
    n = P.squeeze(-2)[:, 2]                                                                         # Ns

    #### Compute the vanishing points of the capsules
    x0 = fz * (l - (l).roll(-1, 0)).divide(n - (n).roll(-1, 0)) # Ns
    y0 = fz * (m - (m).roll(-1, 0)).divide(n - (n).roll(-1, 0)) # Ns

    #### Compute the coefficients of the equation of the ellipse
    A = (l).square() - k_sq                                                                         # Ns
    B = 2*(l).multiply(m)                                                                           # Ns
    C = (m).square() - k_sq                                                                         # Ns
    D = 2*fz*(l).multiply(n)                                                                        # Ns
    E = 2*fz*(m).multiply(n)                                                                        # Ns
    F = (fz**2)*((n).square() - k_sq)                                                               # Ns
    
    #### Compute SDF and occupancy of ellipses and finally the rendered image
    sdf_ellipses = ((A*(x**2) + B*(x*y) + C*(y**2) + D*(x) + E*(y) + F)).max(-1)[0]
    I_ellipses = torch.sigmoid(tau_ellipse * sdf_ellipses.view(img_size,img_size))

    #### Compute intermediate variables for rendering capsule
    temp1 = (B.multiply(x0) - 2*C.multiply(y0) - E) # Ns
    temp2 = B.multiply(y0) + D # Ns
    temp3 = C.multiply(y0.square()) + E.multiply(y0) + F # Ns

    #### Ignore the numerically unstable capsules
    sol_exists =    (torch.isfinite(temp1)) * \
                    (torch.isfinite(temp2)) * \
                    (torch.isfinite(temp3)) * \
                    (temp1.abs() < temp1_max) * \
                    (temp2.abs() < temp2_max) * \
                    (temp3.abs() < temp3_max) * \
                    (n > eps_z_pos) * \
                    (n.roll(-1, 0) > eps_z_pos) * \
                    ((n - n.roll(-1, 0)).abs() > eps_render)

    #### Move ahead with only the numerically stable capsules
    A = A[sol_exists]
    B = B[sol_exists]
    C = C[sol_exists]
    D = D[sol_exists]
    E = E[sol_exists]
    F = F[sol_exists]
    x0 = x0[sol_exists]
    y0 = y0[sol_exists]
    temp1 = temp1[sol_exists]
    temp2 = temp2[sol_exists]
    temp3 = temp3[sol_exists]

    #### Compute the coefficients for the equation of slope of the tangent to the ellipse
    alpha = (   temp1.square() -   4*C.multiply(x0).multiply(temp2)  -   4*C.multiply(temp3 )  -   4*A.multiply(C).multiply(x0.square()) + 4*B*(2*C*x0*y0 + E*x0)  ) # Ns
    beta = (2*(-temp1).multiply(temp2)) - 4*B.multiply(temp3 ) +4*A.multiply(2*C.multiply(x0).multiply(y0) + E.multiply(x0)) # Ns
    gamma = temp2.square() - 4*A.multiply(temp3) # Ns
    discriminant_sq = (beta.square() - 4*alpha.multiply(gamma))

    #### Ignore other types of numerically unstable capsules
    sol_exists =    (discriminant_sq > discriminant_sq_min) * \
                    (alpha.abs() > eps_render) * \
                    (alpha.abs() < alpha_max) * \
                    (beta.abs() < beta_max) * \
                    (gamma.abs() < gamma_max)

    #### Move ahead with only the numerically stable capsules
    A = A[sol_exists]
    B = B[sol_exists]
    C = C[sol_exists]
    D = D[sol_exists]
    E = E[sol_exists]
    F = F[sol_exists]
    x0 = x0[sol_exists]
    y0 = y0[sol_exists]
    alpha = alpha[sol_exists]
    beta = beta[sol_exists]
    gamma = gamma[sol_exists]
    temp1 = temp1[sol_exists]
    temp2 = temp2[sol_exists]
    temp3 = temp3[sol_exists]

    #### Compute the slopes of the tangent to the ellipses
    discriminant = (beta.square() - 4*alpha.multiply(gamma)).sqrt() # Ns
    m1 = (-beta + discriminant).divide(2*alpha) # Ns
    m2 = (-beta - discriminant).divide(2*alpha) # Ns

    # First quadrilateral point which belongs to the first ellipse
    x1 = (-E.multiply(m1) + B.multiply(m1).multiply(x0) - B.multiply(y0) + 2*C.multiply(m1.square()).multiply(x0) - 2*C.multiply(m1).multiply(y0) - D).divide(2*(A+B.multiply(m1) + C.multiply(m1.square()))) # Ns
    y1 = m1.multiply(x1 - x0) + y0 # Ns
    # Second quadrilateral point which belongs to the first ellipse
    x2 = (-E.multiply(m2) + B.multiply(m2).multiply(x0) - B.multiply(y0) + 2*C.multiply(m2.square()).multiply(x0) - 2*C.multiply(m2).multiply(y0) - D).divide(2*(A+B.multiply(m2) + C.multiply(m2.square()))) # Ns
    y2 = m2.multiply(x2 - x0) + y0 # Ns
    # Third quadrilateral point which belongs to the second ellipse
    x3 = (-E.roll(-1,0).multiply(m1) + B.roll(-1,0).multiply(m1).multiply(x0) - B.roll(-1,0).multiply(y0) + 2*C.roll(-1,0).multiply(m1.square()).multiply(x0) - 2*C.roll(-1,0).multiply(m1).multiply(y0) - D.roll(-1,0)).divide(2*(A.roll(-1,0)+B.roll(-1,0).multiply(m1) + C.roll(-1,0).multiply(m1.square()))) # Ns
    y3 = m1.multiply(x3 - x0) + y0 # Ns
    # Fourth quadrilateral point which belongs to the second ellipse
    x4 = (-E.roll(-1,0).multiply(m2) + B.roll(-1,0).multiply(m2).multiply(x0) - B.roll(-1,0).multiply(y0) + 2*C.roll(-1,0).multiply(m2.square()).multiply(x0) - 2*C.roll(-1,0).multiply(m2).multiply(y0) - D.roll(-1,0)).divide(2*(A.roll(-1,0)+B.roll(-1,0).multiply(m2) + C.roll(-1,0).multiply(m2.square()))) # Ns
    y4 = m2.multiply(x4 - x0) + y0 # Ns

    #### Unsqueeze to make an axis for the pixels    
    x0 = x0.unsqueeze(-2)
    y0 = y0.unsqueeze(-2)
    x1 = x1.unsqueeze(-2)
    x2 = x2.unsqueeze(-2)
    x3 = x3.unsqueeze(-2)
    x4 = x4.unsqueeze(-2)
    y1 = y1.unsqueeze(-2)
    y2 = y2.unsqueeze(-2)
    y3 = y3.unsqueeze(-2)
    y4 = y4.unsqueeze(-2)

    #### Unsqueeze to make an axis for the pixels    
    # A = A.unsqueeze(0)
    # B = B.unsqueeze(0)
    # C = C.unsqueeze(0)
    # D = D.unsqueeze(0)
    # E = E.unsqueeze(0)
    # F = F.unsqueeze(0)

    #### Compute the distance function and orientations for the line segments of the quadrilateral
    sdf_line1 = ((x3 - x1)*(y - y1) - (y3 - y1)*(x - x1)).divide(((x3 - x1).square() + (y3 - y1).square()).sqrt())
    sdf_line2 = ((x4 - x3)*(y - y3) - (y4 - y3)*(x - x3)).divide(((x4 - x3).square() + (y4 - y3).square()).sqrt())
    sdf_line3 = ((x2 - x4)*(y - y4) - (y2 - y4)*(x - x4)).divide(((x2 - x4).square() + (y2 - y4).square()).sqrt())
    sdf_line4 = ((x1 - x2)*(y - y2) - (y1 - y2)*(x - x2)).divide(((x1 - x2).square() + (y1 - y2).square()).sqrt())
    orientation = (x3-x1)*(y2-y1) - (y3-y1)*(x2-x1)

    #### Compute the signed distance function for the line segments and then quadrilateral of the capsule
    sdf_lines = torch.sign(orientation).unsqueeze(-1) * torch.stack([sdf_line1, sdf_line2, sdf_line3, sdf_line4],-1)
    sdf_quad = sdf_lines.min(-1)[0]
    sdf_pix = sdf_quad.max(-1)[0]
    I_quadrilateral = torch.sigmoid(tau_quadrilateral * sdf_pix.view(img_size,img_size))

    #### Compute the rendering of the capsules
    I_rendered = torch.maximum(I_quadrilateral,I_ellipses) 
    return I_rendered  
