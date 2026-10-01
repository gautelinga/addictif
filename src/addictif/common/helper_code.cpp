#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Vertex.h>

class Tet {
public:
  Tet(const dolfin::Cell cell){
    for (dolfin::VertexIterator v(cell); !v.end(); ++v)
    {
        const std::size_t pos = v.pos();
        xx_[pos] = v->x(0);
        yy_[pos] = v->x(1);
        zz_[pos] = v->x(2);
    }

    double j11 = xx_[1]-xx_[0];
    double j12 = yy_[1]-yy_[0];
    double j13 = zz_[1]-zz_[0];
    double j21 = xx_[2]-xx_[0];
    double j22 = yy_[2]-yy_[0];
    double j23 = zz_[2]-zz_[0];
    double j31 = xx_[3]-xx_[0];
    double j32 = yy_[3]-yy_[0];
    double j33 = zz_[3]-zz_[0];

    g2x_ = j22*j33-j23*j32;  g3x_ = j13*j32-j12*j33;  g4x_ = j12*j23-j13*j22;
    g2y_ = j23*j31-j21*j33;  g3y_ = j11*j33-j13*j31;  g4y_ = j13*j21-j11*j23;
    g2z_ = j21*j32-j22*j31;  g3z_ = j12*j31-j11*j32;  g4z_ = j11*j22-j12*j21;
    double det = j11 * g2x_ + j12 * g2y_ + j13 * g2z_;
    double d = 1.0/det;
    g2x_ *= d;  g3x_ *= d;  g4x_ *= d;
    g2y_ *= d;  g3y_ *= d;  g4y_ *= d;
    g2z_ *= d;  g3z_ *= d;  g4z_ *= d;
    g1x_ = -g2x_-g3x_-g4x_;  g1y_ = -g2y_-g3y_-g4y_;  g1z_ = -g2z_-g3z_-g4z_;
  }
  void linearbasis(double r,
                   double s,
                   double t,
                   double u,
                   std::vector<double> &N) const
  {
    N[0] = r;
    N[1] = s;
    N[2] = t;
    N[3] = u;
  }
  
  void linearderiv(std::vector<double> &Nx,
                   std::vector<double> &Ny,
                   std::vector<double> &Nz) const {
    Nx[0] = g1x_;
    Nx[1] = g2x_;
    Nx[2] = g3x_;
    Nx[3] = g4x_;

    Ny[0] = g1y_;
    Ny[1] = g2y_;
    Ny[2] = g3y_;
    Ny[3] = g4y_;

    Nz[0] = g1z_;
    Nz[1] = g2z_;
    Nz[2] = g3z_;
    Nz[3] = g4z_;
  }
private:
  std::array<double, 4> xx_, yy_, zz_;
  double g1x_, g1y_, g1z_;
  double g2x_, g2y_, g2z_;
  double g3x_, g3y_, g3z_;
  double g4x_, g4y_, g4z_;

  //static constexpr std::array<int, 10> perm_ = {-1, -1, -1, -1, 9, 6, 8, 7, 5, 4};
};

class AbsVecCell : public dolfin::Expression
{
public:

  // Create expression with 1 component
  AbsVecCell() : dolfin::Expression() {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& ufc_cell) const override
  {
    const uint cell_index = ufc_cell.index;
    const dolfin::Cell dolfin_cell(*u->function_space()->mesh(), cell_index);
    const dolfin::FiniteElement element = *u->function_space()->element();

    std::vector<double> coordinate_dofs;
    dolfin_cell.get_coordinate_dofs(coordinate_dofs);

    const size_t ncoeff = element.space_dimension()/3;

    std::vector<double> coefficients_(3*ncoeff);

    u->restrict(coefficients_.data(), element, dolfin_cell,
                coordinate_dofs.data(), ufc_cell);

    std::vector<double> N_(ncoeff);
    std::fill(N_.begin(), N_.end(), 1./ncoeff);

    double ux = std::inner_product(N_.begin(), N_.end(), coefficients_.begin(), 0.0);
    double uy = std::inner_product(N_.begin(), N_.end(), &coefficients_[1*ncoeff], 0.0);
    double uz = std::inner_product(N_.begin(), N_.end(), &coefficients_[2*ncoeff], 0.0);

    values[0] = sqrt(ux * ux + uy * uy + uz * uz);
  }
  std::shared_ptr<dolfin::Function> u;
};

class AbsGrad : public dolfin::Expression
{
public:

  // Create expression with 1 component
  AbsGrad() : dolfin::Expression() {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& ufc_cell) const override
  {
    const uint cell_index = ufc_cell.index;
    const dolfin::Cell dolfin_cell(*a->function_space()->mesh(), cell_index);
    //dolfin_cell.get_cell_data(ufc_cell);
    const dolfin::FiniteElement element = *a->function_space()->element();

    std::vector<double> coordinate_dofs;
    dolfin_cell.get_coordinate_dofs(coordinate_dofs);
    // const size_t dim = 3; // a->function_space()->mesh()->geometry().dim();
    const size_t ncoeff = 4;
 
    std::vector<double> coefficients_(ncoeff);

    a->restrict(coefficients_.data(), element, dolfin_cell,
                coordinate_dofs.data(), ufc_cell);

    std::vector<double> Nx_(ncoeff);
    std::vector<double> Ny_(ncoeff);
    std::vector<double> Nz_(ncoeff);

    Tet tet(dolfin_cell);
    tet.linearderiv(Nx_, Ny_, Nz_);

    double dadx = std::inner_product(Nx_.begin(), Nx_.end(), coefficients_.begin(), 0.0);
    double dady = std::inner_product(Ny_.begin(), Ny_.end(), coefficients_.begin(), 0.0);
    double dadz = std::inner_product(Nz_.begin(), Nz_.end(), coefficients_.begin(), 0.0);

    values[0] = sqrt(dadx * dadx + dady * dady + dadz * dadz);
  }
  std::shared_ptr<dolfin::Function> a;
};

class Grad : public dolfin::Expression
{
public:

  // Create expression with 3 components
  Grad() : dolfin::Expression(3) {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& ufc_cell) const override
  {
    const uint cell_index = ufc_cell.index;
    const dolfin::Cell dolfin_cell(*a->function_space()->mesh(), cell_index);
    //dolfin_cell.get_cell_data(ufc_cell);
    const dolfin::FiniteElement element = *a->function_space()->element();

    std::vector<double> coordinate_dofs;
    dolfin_cell.get_coordinate_dofs(coordinate_dofs);
    // const size_t dim = 3; // a->function_space()->mesh()->geometry().dim();
    const size_t ncoeff = 4;
 
    std::vector<double> coefficients_(ncoeff);

    a->restrict(coefficients_.data(), element, dolfin_cell,
                coordinate_dofs.data(), ufc_cell);

    std::vector<double> Nx_(ncoeff);
    std::vector<double> Ny_(ncoeff);
    std::vector<double> Nz_(ncoeff);

    Tet tet(dolfin_cell);
    tet.linearderiv(Nx_, Ny_, Nz_);

    double dadx = std::inner_product(Nx_.begin(), Nx_.end(), coefficients_.begin(), 0.0);
    double dady = std::inner_product(Ny_.begin(), Ny_.end(), coefficients_.begin(), 0.0);
    double dadz = std::inner_product(Nz_.begin(), Nz_.end(), coefficients_.begin(), 0.0);

    values[0] = dadx;
    values[1] = dady;
    values[2] = dadz;
  }
  std::shared_ptr<dolfin::Function> a;
};

class Flux : public dolfin::Expression
{
public:

  // Create expression with 1 component
  Flux() : dolfin::Expression() {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& ufc_cell) const override
  {
    const uint cell_index = ufc_cell.index;
    const dolfin::Cell dolfin_cell(*a->function_space()->mesh(), cell_index);
    //dolfin_cell.get_cell_data(ufc_cell);
    const dolfin::FiniteElement element = *a->function_space()->element();
    const dolfin::FiniteElement element_n = *n->function_space()->element();


    std::vector<double> coordinate_dofs;
    dolfin_cell.get_coordinate_dofs(coordinate_dofs);

    std::vector<double> n_coefficients(element_n.space_dimension());
    n->restrict(n_coefficients.data(), element_n, dolfin_cell,
                coordinate_dofs.data(), ufc_cell);
    double nx = n_coefficients[0];
    double ny = n_coefficients[1];
    double nz = n_coefficients[2];

    // const size_t dim = 3; // a->function_space()->mesh()->geometry().dim();
    const size_t ncoeff = 4;
 
    std::vector<double> coefficients_(ncoeff);

    a->restrict(coefficients_.data(), element, dolfin_cell,
                coordinate_dofs.data(), ufc_cell);

    std::vector<double> Nx_(ncoeff);
    std::vector<double> Ny_(ncoeff);
    std::vector<double> Nz_(ncoeff);

    Tet tet(dolfin_cell);
    tet.linearderiv(Nx_, Ny_, Nz_);

    double dadx = std::inner_product(Nx_.begin(), Nx_.end(), coefficients_.begin(), 0.0);
    double dady = std::inner_product(Ny_.begin(), Ny_.end(), coefficients_.begin(), 0.0);
    double dadz = std::inner_product(Nz_.begin(), Nz_.end(), coefficients_.begin(), 0.0);

    values[0] = -(nx*dadx + ny*dady + nz*dadz);
  }
  std::shared_ptr<dolfin::Function> n;
  std::shared_ptr<dolfin::Function> a;
};

class CellSize : public dolfin::Expression
{
public:

  // Create expression with 1 component
  CellSize() : dolfin::Expression() {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& ufc_cell) const override
  {
    const uint cell_index = ufc_cell.index;
    const dolfin::Cell dolfin_cell(*mesh, cell_index);

    values[0] = dolfin_cell.h();
  }
  std::shared_ptr<dolfin::Mesh> mesh;
};

class ScalarDG0 : public dolfin::Expression
{
public:

  // Create expression with 1 component
  ScalarDG0() : dolfin::Expression() {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& ufc_cell) const override
  {
    const uint cell_index = ufc_cell.index;
    const dolfin::Cell dolfin_cell(*a->function_space()->mesh(), cell_index);
    //dolfin_cell.get_cell_data(ufc_cell);
    const dolfin::FiniteElement element = *a->function_space()->element();

    std::vector<double> coordinate_dofs;
    dolfin_cell.get_coordinate_dofs(coordinate_dofs);
    // const size_t dim = 3; // a->function_space()->mesh()->geometry().dim();
    const size_t ncoeff = 4;
 
    std::vector<double> coefficients_(ncoeff);

    a->restrict(coefficients_.data(), element, dolfin_cell,
                coordinate_dofs.data(), ufc_cell);

    std::vector<double> N_(ncoeff);
    for (int i=0; i < ncoeff; ++i)
      N_[i] = 1./ncoeff;

    values[0] = std::inner_product(N_.begin(), N_.end(), coefficients_.begin(), 0.0);
  }
  std::shared_ptr<dolfin::Function> a;
};

int mark_for_refinement(std::shared_ptr<dolfin::MeshFunction<bool>> cell_marker, std::shared_ptr<dolfin::Function> ind, const double tol) {
  std::shared_ptr<const dolfin::Mesh> mesh = ind->function_space()->mesh();
  const dolfin::FiniteElement& element = *ind->function_space()->element();
  assert(element.space_dimension() == 1);
  cell_marker->set_all(false);
  int num_marked = 0;
  for (std::size_t i = 0; i < mesh->num_cells(); ++i)
  {
    dolfin::Cell dolfin_cell(*mesh, i);
    ufc::cell ufc_cell;

    std::vector<double> coefficients_(1);
    std::vector<double> coordinate_dofs_;
    dolfin_cell.get_coordinate_dofs(coordinate_dofs_);

    ind->restrict(coefficients_.data(), element, dolfin_cell,
                  coordinate_dofs_.data(), ufc_cell);

    // std::cout << coefficients_[0] << std::endl;
    bool flag = coefficients_[0] > tol;
    cell_marker->set_value(i, flag);
    if (flag) ++num_marked;
  }
  return num_marked;
}

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<AbsGrad, std::shared_ptr<AbsGrad>, dolfin::Expression>
    (m, "AbsGrad")
    .def(py::init<>())
    .def_readwrite("a", &AbsGrad::a);
  py::class_<Grad, std::shared_ptr<Grad>, dolfin::Expression>
    (m, "Grad")
    .def(py::init<>())
    .def_readwrite("a", &Grad::a);
  py::class_<Flux, std::shared_ptr<Flux>, dolfin::Expression>
    (m, "Flux")
    .def(py::init<>())
    .def_readwrite("n", &Flux::n)
    .def_readwrite("a", &Flux::a);
  py::class_<ScalarDG0, std::shared_ptr<ScalarDG0>, dolfin::Expression>
    (m, "ScalarDG0")
    .def(py::init<>())
    .def_readwrite("a", &ScalarDG0::a);
  py::class_<CellSize, std::shared_ptr<CellSize>, dolfin::Expression>
    (m, "CellSize")
    .def(py::init<>())
    .def_readwrite("mesh", &CellSize::mesh);
  py::class_<AbsVecCell, std::shared_ptr<AbsVecCell>, dolfin::Expression>
    (m, "AbsVecCell")
    .def(py::init<>())
    .def_readwrite("u", &AbsVecCell::u);
  m.def("mark_for_refinement", &mark_for_refinement, py::arg("cell_marker"), py::arg("ind"), py::arg("tol"));
}