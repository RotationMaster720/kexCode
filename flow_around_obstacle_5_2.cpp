/**
 * @file flow_around_obstacle.cpp
 * @author Sebastian Myrbäck
 * @brief Solve the Navier-Stokes obstacle fictitious domain problem with the Taylor-Hood finite element pair
 * using Nitsche's method for imposing Dirichlet boundary conditions.
 * @date 2026-01-30
 */

// Dependencies
#include "../cutfem.hpp"
#include <filesystem>
#include <fstream>

using mesh_t       = MeshQuad2;
using funtest_t    = TestFunction<mesh_t>;
using fct_t        = FunFEM<mesh_t>;
using activemesh_t = ActiveMesh<mesh_t>;
using time_interface_t = TimeInterface<mesh_t>;
using space_t      = GFESpace<mesh_t>;
using cutspace_t   = CutFESpace<mesh_t>;
using lagrange_t   = LagrangeQuad2;
using matrix_t     = std::map<std::pair<int, int>, double>;
using real         = algoim::real;
using Vec2         = algoim::uvector<real,2>;

using namespace globalVariable;


namespace ex1 {

    std::string problem_name = "channel-5_2";

    double nu = 1e-3;
    double nuinv = 1./nu;
    double rho = 1.0;

    const double boundary_penalty   = 35.;
    const double interior_penalty   = 100.;
    const double ghost_penalty_u    = 1e1;
    const double ghost_penalty_p    = 1e1;

    // Background domain
    double bottom_left_x = 0; // Bottom left corner x coordinate
    double bottom_left_y = 0; // Bottom left corner y coordinate
    double width_domain  = 2.2; // Width of the domain
    double height_domain = 0.41; // Height of the domain

    const double T = 15; // 40
    const double um = 1;

    double radius = 0.1;
    double center_x = 0.2;
    double center_y = 0.2;
    double A = 0;
    double omega = 1;

    double fun_levelset(double* P, int i, const double t) {
        const double x = P[0], y = P[1];
        return -(((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)) - radius*radius);
    }

    struct Levelset {
        // Algoim always integrates over the negative part of the levelset function so for this example,
        // we define the levelset function as the negative of the one defined above

        double t;

        template <typename V> typename V::value_type operator()(const V &P) const {
            
            auto xc = center_x;
            auto yc = center_y;
            return -((P[0]-xc)*(P[0]-xc) + (P[1]-yc)*(P[1]-yc) - radius*radius - 1e-14);
        }

        template <typename T> algoim::uvector<T, 2> grad(const algoim::uvector<T, 2> &x) const {
            auto xc = center_x + A*std::sin(omega*t);
            auto yc = center_y;
            return algoim::uvector<T, 2>(-2*(x(0)-xc), -2*(x(1)-yc));
        }

        R2 normal(std::span<double> P) const {
            double xc = center_x + A*std::sin(omega*t);
            double yc = center_y;
            double norm = std::sqrt(std::pow(2*(P[0]-xc), 2) + std::pow(2*(P[1]-yc), 2));

            return R2(-2*(P[0]-xc) / norm, -2*(P[1]-yc) / norm);
        }
    };

    double fun_normal(double* P, int i, double t) {
        if (i == 0)
            return -(P[0] - center_x - A*std::sin(omega*t)) / std::sqrt((P[0] - center_x - A*std::sin(omega*t)) * (P[0] - center_x - A*std::sin(omega*t)) + (P[1] - center_y)*(P[1] - center_y));
        else
            return -(P[1] - center_y) / std::sqrt((P[0] - center_x - A*std::sin(omega*t)) * (P[0] - center_x - A*std::sin(omega*t)) + (P[1] - center_y)*(P[1] - center_y));
    }

    double rhs(double *P, int component, double t) {
        double x = P[0];
        double y = P[1];
        
        if (component == 0)
            return std::pow(height_domain/2,2) - std::pow(y,2);
            //return std::sin(M_PI*y)*(std::cos(M_PI*y)*std::sin(M_PI*x)*std::sin(M_PI*x)*std::cos(t) + M_PI*std::cos(M_PI*x)*std::cos(M_PI*x)*std::cos(M_PI*y)*std::sin(t) - M_PI*std::cos(M_PI*y)*std::sin(M_PI*x)*std::sin(M_PI*x)*std::sin(t) + M_PI*std::cos(M_PI*x)*std::sin(M_PI*x)*std::sin(M_PI*x)*std::sin(M_PI*x)*std::sin(M_PI*y)*std::sin(M_PI*y)*std::sin(M_PI*y)*std::sin(t)*std::sin(t) - 2*nu*M_PI*M_PI*std::cos(M_PI*x)*std::cos(M_PI*x)*std::cos(M_PI*y)*std::sin(t) + 6*nu*M_PI*M_PI*std::cos(M_PI*y)*std::sin(M_PI*x)*std::sin(M_PI*x)*std::sin(t) + M_PI*std::cos(M_PI*x)*std::cos(M_PI*y)*std::cos(M_PI*y)*std::sin(M_PI*x)*std::sin(M_PI*x)*std::sin(M_PI*x)*std::sin(M_PI*y)*std::sin(t)*std::sin(t));
        else
            return 0;
            //return std::sin(M_PI*x)*(M_PI*std::cos(M_PI*x)*std::cos(M_PI*y)*std::cos(M_PI*y)*std::sin(t) - std::cos(M_PI*x)*std::sin(M_PI*y)*std::sin(M_PI*y)*std::cos(t) - M_PI*std::cos(M_PI*x)*std::sin(M_PI*y)*std::sin(M_PI*y)*std::sin(t) + M_PI*std::cos(M_PI*y)*std::sin(M_PI*x)*std::sin(M_PI*x)*std::sin(M_PI*x)*std::sin(M_PI*y)*std::sin(M_PI*y)*std::sin(M_PI*y)*std::sin(t)*std::sin(t) + 2*nu*M_PI*M_PI*std::cos(M_PI*x)*std::cos(M_PI*y)*std::cos(M_PI*y)*std::sin(t) - 6*nu*M_PI*M_PI*std::cos(M_PI*x)*std::sin(M_PI*y)*std::sin(M_PI*y)*std::sin(t) + M_PI*std::cos(M_PI*x)*std::cos(M_PI*x)*std::cos(M_PI*y)*std::sin(M_PI*x)*std::sin(M_PI*y)*std::sin(M_PI*y)*std::sin(M_PI*y)*std::sin(t)*std::sin(t));

    }

    double bc_fun(double* P, int component, double t) {
        double x = P[0];
        double y = P[1];
        
        if (component == 0 && std::abs(x-0) < 0.000001)
            return 4*1.5*y*(0.41-y)/std::pow(0.41,2);
        else
            return 0;
    }

    //noslip condition
    double noslipfun(double* P, int component, double t) {
        double x = P[0];
        double y = P[1];
        
        return 0;
    }


    double pexact(double* P, int component, double t) {
        double x = P[0];
        double y = P[1];
        double c = 0.;

        return -2*nu*(x-width_domain);
        //return std::cos(y*M_PI)*std::sin(t)*std::sin(x*M_PI)*std::cos(x*M_PI)*std::sin(y*M_PI);
    }

    double pexact_d(double* P, int component, int domain, double t) {
        return pexact(P, component, t);
    }

    double onefunc(double* P, int component) {
        return 1.;
    }
    
    double fun_normal_obstacle(double* P, int i, double t) {
        if (i == 0)
            return (P[0] - center_x)/ std::sqrt((P[0]- center_x) * (P[0]- center_x) + (P[1]- center_y)*(P[1]- center_y));
        else
            return (P[1] - center_y)/ std::sqrt((P[0]- center_x) * (P[0]- center_x) + (P[1]- center_y)*(P[1]- center_y));
    }

    double fun_tangent_obstacle(double *P, int i, double t) {
        if (i == 0)
            return fun_normal_obstacle(P, 1, t);
        else
            return -fun_normal_obstacle(P, 0, t);
    }
}



std::vector<const GTypeOfFE<mesh_t> *> L2_space = {&DataFE<mesh_t>::P0, &DataFE<mesh_t>::P1dc, &DataFE<mesh_t>::P2dc, &DataFE<mesh_t>::P3dc};
std::vector<const GTypeOfFE<mesh_t> *> H1_space = {&DataFE<mesh_t>::P0, &DataFE<mesh_t>::P1, &DataFE<mesh_t>::P2, &DataFE<mesh_t>::P3};
std::vector<const GTypeOfFE<Mesh1> *>  FE_time  = {&DataFE<Mesh1>::P0Poly, &DataFE<Mesh1>::P1Poly, &DataFE<Mesh1>::P2Poly,
                                                   &DataFE<Mesh1>::P3Poly};




#define EX1


#if defined(EX1)
    using namespace ex1;
#endif

std::string fe_pair;
std::string bc_method;

using algoimcutfem_t = AlgoimCutFEM<mesh_t, Levelset>;

int main(int argc, char **argv) {
    MPIcf cfMPI(argc, argv);

    // User-defined parameters
    const int k                 = 2;  // degree of velocity space
    const int n                 = 1;  // degree of time space
    const int mesh_refinements  = 1;
    
    // Mesh and time step 
    double h0 = 1./(std::sqrt(2)*2); // initial mesh size
    double h  = h0/8;
    double dT = 0.05; //h/h0;
    
    const double cfl = um * dT / h; 

    // For later
    std::list<int> inflow = {4};
    std::list<int> walls  = {1,3};
    std::list<int> outflow= {2};
    std::list<int> inflow_and_walls = {4,1,3};


    // Space quadrature
    ProblemOption option;
    // option.order_space_element_quadrature_ = (k > 1) ? k*k+2 : 5;
    const int quadrature_order_space = (3*k + 2) / 2; // integer ceil of (3k+1)/2 for integers
    option.order_space_element_quadrature_ = quadrature_order_space;
    option.clear_matrix_ = true;

    // Time integration quadrature
    // const int quadrature_order_time = (n > 1) ? n*n+2 : 5;
    const int quadrature_order_time = (3*n + 2) / 2; 
    const QuadratureFormular1d &q_time(*Lobatto(quadrature_order_time)); // specify order of quadrature in time
    const Uint n_quad_pts_time   = q_time.n;   // number of quadrature points per time interval 
    const Uint last_quad_pt_time = n_quad_pts_time - 1;


    // Data paths and containers
    std::string path_output_data("./output_files/navier_stokes/" + problem_name + "/data/");
    std::string path_output_figures("./output_files/navier_stokes/" + problem_name + "/paraview/");     

    if (MPIcf::IamMaster()) {
        std::filesystem::create_directories(path_output_data);
        std::filesystem::create_directories(path_output_figures);
    }

    std::array<double, mesh_refinements> L2_errors_u, L2_errors_p, L2L2_errors_u, L2L2_errors_p, conv_L2_u, conv_L2L2_u, conv_H1_u, conv_L2_p, conv_L2L2_p, conv_L2_l, boundary_error_u, normal_boundary_error_u, conv_boundary, conv_normal_boundary, pwise_div, grad_errors_u, H1_errors_u, H1_errors_p, hs, dts, numb_elems, numb_dofs;

    const std::string solver_name("mumps");
    const std::string bc_method("nitsche");

    fe_pair = "Taylor-Hood";
    
    Levelset phi;
    
    std::cout << "\n Running Navier-Stokes with " << fe_pair << " elements of order (" << k << ", " << k-1 << ") and with quadrature points (" << quadrature_order_space << ", " << quadrature_order_time << ") in space and time \n \n";

    int num_time_steps = int(T / dT);
    for (int j = 0; j < mesh_refinements; ++j) {
        int nx = (int)(width_domain / h) + 1;
        int ny = (int)(height_domain / h) + 1;
                
        Mesh1 TimeDiscretization(num_time_steps + 1, 0., T);
        // std::cout << "\n Running Navier-Stokes with " << typeid(Mesh1).name() << endl;
        FESpace1 Ih(TimeDiscretization, *FE_time.at(n));
        FESpace1 Ih_interpolation(TimeDiscretization, *FE_time.at(3));
        const int ndf_time_slab = Ih[0].NbDoF();

        Normal n;
        Tangent t;

        std::cout << "--------------------------------------------" << '\n';
        std::cout << "--------------------------------------------" << '\n';
        std::cout << "Mesh refinement " << j + 1 << "/" << mesh_refinements << '\n';
        std::cout << "h  = " << h << ", dT = " << dT << '\n';

        // Mesh
        mesh_t Th(nx, ny, bottom_left_x, bottom_left_y, width_domain, height_domain);
    

        // P1-interpolated level set function
        space_t Lh(Th, DataFE<mesh_t>::P1);
        std::vector<fct_t> levelsets(n_quad_pts_time);
        // for (auto &levelset : levelsets) {
        //     levelset.init(Lh, fun_levelset);
        // }

        // fct_t phi(Lh, levelset);
        // InterfaceLevelSet<mesh_t> surface(Th, phi);
        time_interface_t surface(q_time);

        // Finite element spaces
        lagrange_t FE_velocity(k);       // Velocity space
        lagrange_t FE_velocity_int(3);   // for interpolation
        space_t V_interpolation(Th, FE_velocity_int);
        space_t P_interpolation(Th, DataFE<mesh_t>::P3);   
        
        // Taylor-Hood element pair
        space_t V(Th, FE_velocity);
        space_t P(Th, *H1_space.at(k-1));
        space_t NSpace(Th, DataFE<mesh_t>::NumberSpaceQ);

        // Matrix to hold non-linear part of the weak form
        matrix_t mat_NL;

        // Problem object
        algoimcutfem_t navier_stokes(q_time, phi, option);

        std::vector<double> lift_vector(num_time_steps), drag_vector(num_time_steps);

        double error_L2L2_uh = 0, error_L2L2_ph = 0, error_LinfL2_ph = 0;
        int iter = 0;
        while (iter < num_time_steps) {

            double current_time = iter * dT;    // first time point in current time interval
            const TimeSlab &In(Ih[iter]);
            const TimeSlab &In_interpolation(Ih_interpolation[iter]);

            std::cout << " -------------------------------------------------------\n";
            std::cout << " -------------------------------------------------------\n";
            std::cout << " Time step \t : \t" << iter + 1 << "/" << num_time_steps << '\n';
            std::cout << " Time      \t : \t" << current_time << '\n';

            // Initialization of the surface in each quadrature point
            for (int i = 0; i < n_quad_pts_time; ++i) {
                const double tt  = In.Pt(R1(q_time(i).x));
                phi.t = tt;
                levelsets[i].init(Lh, fun_levelset, tt);
                surface.init(i, Th, phi);
                
            }


            activemesh_t active_mesh(Th);
            active_mesh.truncate(surface, 1);   // truncate the mesh, removing elements where levelset > 0

            // Cut finite element spaces
            cutspace_t Vh(active_mesh, V);     // Velocity space
            cutspace_t Ph(active_mesh, P);     // Pressure space
            cutspace_t Nhu(active_mesh, NSpace);    // Numberspace to enforce int_Gamma v*n = 0
            cutspace_t Nhp(active_mesh, NSpace);    // Numberspace to enforce int_Gamma p = 0
            
            navier_stokes.initSpace(Vh, In);
            navier_stokes.add(Ph, In);
            // navier_stokes.add(Nhu, In);
            navier_stokes.add(Nhp, In);

            std::cout << "NB_DOFS = " << navier_stokes.get_nb_dof() << '\n';
            
            funtest_t u(Vh, 2), v(Vh, 2);  // du denotes the update delta u
            funtest_t p(Ph, 1), q(Ph, 1);  // dp denotes the update delta p
            // funtest_t xi_u(Nhu, 1), chi_u(Nhu, 1);
            funtest_t xi_p(Nhp, 1), chi_p(Nhp, 1);
            
            // Interpolate exact functions
            fct_t bc(V_interpolation, In_interpolation, bc_fun);
            fct_t p_exact(P_interpolation, In_interpolation, pexact);
            fct_t noslip(V_interpolation, In_interpolation, noslipfun);
            fct_t fh(V_interpolation, In_interpolation, rhs);     // rhs force

            std::vector<double> data_init(navier_stokes.get_nb_dof());
            std::span<double> data_init_span(data_init);
            std::span<double> data_uh0 = std::span<double>(data_init.data(), Vh.NbDoF()); // velocity in first time DOF

            navier_stokes.initialSolution(data_init_span);

            std::vector<double> data_all(data_init);
            int idxp0 = Vh.NbDoF() * In.NbDoF(); // index for when p starts in the array
            std::span<double> data_uh  = std::span<double>(data_all.data(), Vh.NbDoF() * In.NbDoF()); // velocity for all time DOFs
            std::span<double> data_ph  = std::span<double>(data_all.data() + idxp0, Ph.NbDoF() * In.NbDoF());

            fct_t u0(Vh, data_uh0);
            fct_t uh(Vh, In, data_uh);
            fct_t ph(Ph, In, data_ph);

            fct_t normal_obst(V_interpolation, In_interpolation, fun_normal_obstacle);

            fct_t tangent_obst(V_interpolation, In_interpolation, fun_tangent_obstacle);

            int newton_iterations = 0;
            bool newton_ok = false;     // flag for convergence of Newton's method
            while (1) {

                if (newton_iterations == 0) {
                    // Add terms that are in the residual

                    // Weak formulation
                    navier_stokes.addBilinear(
                        + innerProduct(rho*u, v)
                        , active_mesh
                        , 0
                        , In);

                    navier_stokes.addBilinear(
                        + innerProduct(dt(u), rho*v)
                        + contractProduct(nu*rho*grad(u), grad(v))
                        - innerProduct(p, div(v)) 
                        + innerProduct(div(u), q)
                        , active_mesh
                        , In);

                    navier_stokes.addBilinear(
                        - innerProduct(nu*rho*grad(u)*n, v)
                        - innerProduct(u, nu*rho*grad(v)*n) 
                        + innerProduct(boundary_penalty*nu*rho/h*u, v) // boundary_penalty*nu*rho = lambda
                        + innerProduct(boundary_penalty/h*u*n, v*n) // ????
                        + innerProduct(p, v*n)
                        - innerProduct(u * n, q)
                        , surface
                        , In);


                    navier_stokes.addBilinear(
                        - innerProduct(nu*rho * grad(u) * n, v)
                        - innerProduct(u, nu*rho * grad(v) * n) 
                        + innerProduct(boundary_penalty*nu*rho/h * u, v)
                        + innerProduct(boundary_penalty/h * u*n, v*n)
                        + innerProduct(p, v * n)
                        - innerProduct(u * n, q)
                        , active_mesh
                        , INTEGRAL_BOUNDARY
                        , In
                        , inflow_and_walls
                        
                    );
                    // , inflow

                    navier_stokes.addPatchStabilization( 
                        + innerProduct(ghost_penalty_u * nu * rho * std::pow(h, -2) * jump(u), jump(v)) 
                        + innerProduct(ghost_penalty_p * std::pow(h, 0) * jump(p), jump(q)) 
                        , active_mesh
                        , In
                    );

                    navier_stokes.addBilinear(
                        + innerProduct(xi_p, q)
                        + innerProduct(p, chi_p)
                        , active_mesh
                        , In
                    );
                }

                // Add (- right hand side) to the residual
                navier_stokes.addLinear(
                    - innerProduct(u0.exprList(), v)
                    , active_mesh
                    , 0
                    , In);
                
                //
                //navier_stokes.addLinear(
                //    - innerProduct(fh.exprList(), v)
                //    , active_mesh
                //    , In);
                
                // Följande termer är l(v), skriv om från exakt till generllt
                navier_stokes.addLinear(
                    + innerProduct(bc.exprList(), nu*rho*grad(v)*n) 
                    - innerProduct(bc.exprList(), boundary_penalty*nu*rho/h*v)
                    - innerProduct(bc*n, boundary_penalty/h*v*n)
                    + innerProduct(bc*n, q)
                    , active_mesh
                    , INTEGRAL_BOUNDARY
                    , In
                    , inflow_and_walls
                );
                    
                navier_stokes.addLinear(
                    + innerProduct(noslip.exprList(), nu*rho*grad(v)*n) 
                    - innerProduct(noslip.exprList(), boundary_penalty*nu*rho/h*v)
                    - innerProduct(noslip*n, boundary_penalty/h*v*n)
                    + innerProduct(noslip*n, q)
                    , surface
                    , In
                );
                
                // matches the term int_Omega p dx added in the bilinear form
                // forces integral of p to be 0
                navier_stokes.addLinear(
                    - innerProduct(p_exact.expr(), chi_p)
                    , active_mesh
                    , In
                );
                
                
                navier_stokes.addMatMul(data_all); // add matrix*solution to rhs

                // assemble Jacobian in the matrix mat_NL
                navier_stokes.set_map(mat_NL);     
                mat_NL = navier_stokes.mat_;

                funtest_t du1(Vh, 1, 0), du2(Vh, 1, 1), v1(Vh, 1, 0), v2(Vh, 1, 1);
                
                // uh is the velocity solution from the previous Newton iteration
                auto ux = uh.expr(0);
                auto uy = uh.expr(1);
                
                // Linearized advection term
                navier_stokes.addBilinear(
                    + innerProduct(du1 * dx(ux) + du2 * dy(ux), rho*v1) 
                    + innerProduct(du1 * dx(uy) + du2 * dy(uy), rho*v2) 
                    + innerProduct(ux * dx(du1) + uy * dy(du1), rho*v1) 
                    + innerProduct(ux * dx(du2) + uy * dy(du2), rho*v2)
                    , active_mesh
                    , In);

                navier_stokes.addLinear(
                    + innerProduct(ux * dx(ux) + uy * dy(ux), rho*v1) 
                    + innerProduct(ux * dx(uy) + uy * dy(uy), rho*v2)
                    , active_mesh
                    , In);
                
                navier_stokes.solve(mat_NL, navier_stokes.rhs_);

                // std::span<double> duh = std::span<double>(navier_stokes.rhs_.data(), Vh.NbDoF() * In.NbDoF());
                // std::span<double> dph = std::span<double>(navier_stokes.rhs_.data() + Vh.NbDoF() * In.NbDoF(), Ph.NbDoF() * In.NbDoF());
                // std::span<double> dlh = std::span<double>(navier_stokes.rhs_.data() + Vh.NbDoF() * In.NbDoF() + Ph.NbDoF() * In.NbDoF(), Qh.NbDoF() * In.NbDoF());

                std::span<double> dw = std::span<double>(navier_stokes.rhs_.data(), navier_stokes.get_nb_dof());    // solution vector of the linearized system
                
                // Compute the largest in absolute value of the Newton update dw
                auto it = std::max_element(dw.begin(), dw.end(),
                    [](double a, double b) { return std::abs(a) < std::abs(b); });

                // double residual_norm = (it != dw.end()) ? std::abs(*it) : 0.0;
                double residual_norm = std::abs(*it);

                // const double residual_norm = *max_res_dw;
                std::cout << " Residual error \t : \t" << residual_norm << '\n';

                // Get new solution as w = w_old - dw
                std::transform(data_all.begin(), data_all.end(), dw.begin(), data_all.begin(), [](double a, double b) { return a - b; });

                // Clear the rhs vector for the next Newton iteration
                navier_stokes.rhs_.resize(navier_stokes.get_nb_dof());  
                std::fill(navier_stokes.rhs_.begin(), navier_stokes.rhs_.end(), 0.);

                newton_iterations++;


                if (residual_norm < 1e-8) {
                    newton_ok = true;
                    navier_stokes.saveSolution(std::span<double>(data_all));
                    break;
                }
                if (newton_iterations >= 7) break;
            }

            navier_stokes.cleanBuildInMatrix();
            navier_stokes.set_map();
            mat_NL.clear();
            std::fill(navier_stokes.rhs_.begin(), navier_stokes.rhs_.end(), 0.0);

            if (!newton_ok) {
                std::cout << "Newton's method did not converge in 7 iterations, breaking...\n";
                break;
            }

            
            // Compute errors in the last time step
            std::vector<double> vec_uh(Vh.get_nb_dof());
            std::vector<double> vec_ph(Ph.get_nb_dof());

            // Compute the numerical solution evaluated in the end-point of the last time interval
            // to get the solution in T
            for (int n = 0; n < ndf_time_slab; ++n) {
                // get the DOFs of u corresponding to DOF n in time and sum with the
                // previous n to get the solution evaluated in endpoint of the time-interval
                std::vector<double> u_dof_n(data_uh.begin() + n * Vh.get_nb_dof(),
                                            data_uh.begin() + (n + 1) * Vh.get_nb_dof());
                std::transform(vec_uh.begin(), vec_uh.end(), u_dof_n.begin(), vec_uh.begin(), std::plus<double>());

                std::vector<double> p_dof_n(data_ph.begin() + n * Ph.get_nb_dof(),
                                            data_ph.begin() + (n + 1) * Ph.get_nb_dof());
                std::transform(vec_ph.begin(), vec_ph.end(), p_dof_n.begin(), vec_ph.begin(), std::plus<double>());
            }
            
            // FEM functions in the last time instance
            fct_t uh_T(Vh, vec_uh);
            fct_t ph_T(Ph, vec_ph);

            auto uh_0dx = dx(uh_T.expr(0));
            auto uh_1dy = dy(uh_T.expr(1));

    
            double error_L2_ph  = L2_norm_cut(ph_T, pexact_d, In, q_time, last_quad_pt_time, phi, 0, 1);
            double error_div_uh = maxNormCut(uh_0dx + uh_1dy, active_mesh);


            error_L2L2_ph += L2L2_norm(ph, pexact_d, active_mesh, In, q_time, phi);
            

            fct_t p_error(Ph, pexact, current_time + dT);
            std::transform(p_error.v.begin(), p_error.v.end(), ph_T.v.begin(), p_error.v.begin(), std::minus<double>()); 


            std::cout << " --- L2 ERRORS: --- " << '\n';
            std::cout << " ||p-ph||_L2 = " << error_L2_ph << '\n';
            // std::cout << " ||grad(u-uh)||_L2 = " << error_H1_uh << '\n';
            std::cout << " ||div(uh)||_infty = " << error_div_uh << '\n';
            std::cout << "\n";
            
            L2_errors_p[j]   = error_L2_ph;
            pwise_div[j]     = error_div_uh;
            // H1_errors_u[j]   = std::sqrt(error_L2_uh*error_L2_uh + error_H1_uh*error_H1_uh);
            hs[j]            = h;
            dts[j]           = dT;
            numb_elems[j]    = active_mesh.get_nb_element();
            numb_dofs[j]     = navier_stokes.rhs_.size();

            if (j >= 1) {
                conv_L2_u[j] = std::log(L2_errors_u[j] / L2_errors_u[j-1]) / std::log(hs[j] / hs[j - 1]);
                conv_H1_u[j] = std::log(H1_errors_u[j] / H1_errors_u[j-1]) / std::log(hs[j] / hs[j - 1]);
                conv_L2_p[j] = std::log(L2_errors_p[j] / L2_errors_p[j-1]) / std::log(hs[j] / hs[j - 1]);
                conv_boundary[j] = std::log(boundary_error_u[j] / boundary_error_u[j-1]) / std::log(hs[j] / hs[j - 1]);
                conv_normal_boundary[j] = std::log(normal_boundary_error_u[j] / normal_boundary_error_u[j-1]) / std::log(hs[j] / hs[j - 1]);
            }
            
            // Calculate Lift and Drag
            auto ut = uh.expr(0)*tangent_obst.expr(0) + uh.expr(1)*tangent_obst.expr(1);
            auto dutdx = dx(ut);
            auto dutdy = dy(ut);

            // Complete drag formula
            auto dutdn = dutdx * normal_obst.expr(0) + dutdy * normal_obst.expr(1);

            auto drag_integrand = rho * nu * dutdn * normal_obst.expr(1) - ph.expr() * normal_obst.expr(0);
            auto lift_integrand = rho * nu * dutdn * normal_obst.expr(0) + ph.expr() * normal_obst.expr(1);
            double P[2] = {0, 0.41/2};
            double u_bar = bc_fun(P,0,0);
            std::cout << u_bar;
            double drag_force = integral_algoim(drag_integrand, *surface(last_quad_pt_time), 0, phi, In, q_time, last_quad_pt_time);
            double lift_force = -integral_algoim(lift_integrand, *surface(last_quad_pt_time), 0, phi, In, q_time, last_quad_pt_time);
            drag_vector[iter] = 2*drag_force/(0.1*std::pow(2*u_bar/3,2));
            lift_vector[iter] = 2*lift_force/(0.1*std::pow(2*u_bar/3,2));
            // Write solutions to Paraview in each time step if not performing a convergence study
            if (MPIcf::IamMaster() && (mesh_refinements == 1)) {

                // Paraview<mesh_t> background_writer(Th, path_output_figures + "Th" + std::to_string(iter) + ".vtk");
                // background_writer.add(levelsets[0], "levelset_first", 0, 1);
                // background_writer.add(levelsets[last_quad_pt_time], "levelset_last", 0, 1);

                Paraview<mesh_t> paraview(active_mesh, last_quad_pt_time, path_output_figures + "navier_stokes_" + bc_method + std::to_string(iter) + ".vtk");
                paraview.add(levelsets[0], "levelset_first", 0, 1);
                paraview.add(levelsets[last_quad_pt_time], "levelset_last", 0, 1);
                paraview.add(uh, "velocity", 0, 2);
                paraview.add(ph, "pressure", 0, 1);
                paraview.add(fabs(uh_0dx + uh_1dy), "divergence");
                paraview.writeActiveMesh(active_mesh, path_output_figures + "active_mesh_" + std::to_string(iter) + ".vtk");

                const int algoim_domain = -1;   // get neg part of levelset
                const int side = 0;             // not used for algoim_domain = -1
                paraview.writeAlgoimQuadrature(active_mesh, phi, In, q_time, 0, algoim_domain, side, path_output_figures + "AlgoimQuadrature_0_" + std::to_string(iter + 1) + ".vtk");
                paraview.writeAlgoimQuadrature(active_mesh, phi, In, q_time, last_quad_pt_time, algoim_domain, side, path_output_figures + "AlgoimQuadrature_N_" + std::to_string(iter + 1) + ".vtk");
            }
            iter++;
        }

        // Saves drag and lift
        if (MPIcf::IamMaster()) {
            std::string filename = path_output_data +"lift_array_" + std::to_string(numb_dofs[j]) + ".csv";
            std::ofstream file(filename);
            for(int i = 0; i < num_time_steps; i++) {
                file << lift_vector[i];
                if(i < num_time_steps-1) file << "\n";
            }
            file.close();
        

            filename = path_output_data + "drag_array_" + std::to_string(numb_dofs[j]) + ".csv";
            file.open(filename);
            for(int i = 0; i < num_time_steps; i++) {
                file << drag_vector[i];
                if(i < num_time_steps-1) file << ", \n";
            }
            file.close();
        }

        L2L2_errors_u[j] = std::sqrt(error_L2L2_uh);
        L2L2_errors_p[j] = std::sqrt(error_L2L2_ph);
        if (j >= 1) {
            conv_L2L2_u[j] = std::log(L2L2_errors_u[j] / L2L2_errors_u[j-1]) / std::log(hs[j] / hs[j - 1]);
            conv_L2L2_p[j] = std::log(L2L2_errors_p[j] / L2L2_errors_p[j-1]) / std::log(hs[j] / hs[j - 1]);
        }
        

        h *= 0.5;

    }

    std::cout << std::setprecision(16);
    std::cout << '\n';
    std::cout << "L2 error u = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << L2_errors_u.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
    std::cout << "L2 conv  u = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << conv_L2_u.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    std::cout << '\n';
    std::cout << "L2 error p = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << L2_errors_p.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
    std::cout << "L2 conv  p = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << conv_L2_p.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    std::cout << '\n';
    std::cout << "L2L2 error u = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << L2L2_errors_u.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
    std::cout << "L2L2 conv  u = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << conv_L2L2_u.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    std::cout << '\n';
    std::cout << "L2L2 error p = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << L2L2_errors_p.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
    std::cout << "L2L2 conv  p = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << conv_L2L2_p.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
    
    std::cout << '\n';
    std::cout << "H1 error u = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << H1_errors_u.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
    std::cout << "H1 conv  u = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << conv_H1_u.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]";
    std::cout << '\n';


    std::cout << '\n';
    std::cout << "Pointwise divergence = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << pwise_div.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << '\n';
    std::cout << '\n';

    std::cout << '\n';

    std::cout << "Boundary error u = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << boundary_error_u.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
    std::cout << "Boundary conv  u = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << conv_boundary.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]";
    std::cout << '\n';
    std::cout << '\n';

    std::cout << "Boundary normal error u = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << normal_boundary_error_u.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
    std::cout << "Boundary normal conv  u = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << conv_normal_boundary.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]";
    std::cout << '\n';
    std::cout << '\n';


    std::cout << "h = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << hs.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << '\n';
    std::cout << '\n';
    std::cout << "Number of mesh elements = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << numb_elems.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << '\n';
    
    std::cout << "Number of degrees of freedom = [";
    for (int i = 0; i < mesh_refinements; i++) {

        std::cout << numb_dofs.at(i);
        if (i < mesh_refinements - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << '\n';

    return 0;
}

