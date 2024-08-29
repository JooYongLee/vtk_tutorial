
#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>


bool testabc()
{}

//#include "PCA.hpp"
#include "pca.h"
using namespace Eigen;
using namespace std;
bool compare(std::pair<int, double> p1, std::pair<int, double> p2) { return p1.second - p2.second; }
void test()
{





















	Eigen::MatrixXd m(2, 2);
	m(0, 1) = 1;
	m(0, 0) = 0.5;
	m(1, 0) = 0.5;
	// Matrisin aritmetik ortalamasi
	std::cout << m.mean() << std::endl;
	// Matris sutunlarinin aritmetik ortalamasi
	std::cout << "m.colwise().mean()" << m.colwise().mean() << std::endl;
	// 0.25, 0.5

	
	Eigen::VectorXd mean_vector = m.colwise().mean();
	// Merkezleme
	// Amacimiz sutunlardan sutun ortalamalarini cikarmak:
	// 0.25     0.5
	// -0.25    -0.5
	std::cout << m << std::endl;
	std::cout << mean_vector << std::endl;
	std::cout << m.rowwise() - mean_vector.transpose() << std::endl;
	Eigen::MatrixXd centered = m.rowwise() - mean_vector.transpose();

	// Kovaryans Matrisi Cikarimi
	Eigen::MatrixXd cov = (centered.adjoint() * centered) / (m.rows() - 1);
	std::cout << "cov" <<cov << std::endl;

	// Eigenvector ayristirmasi
	Eigen::SelfAdjointEigenSolver<MatrixXd> eig_solver(2);

	eig_solver.compute(cov);
	std::cout << "eig_solver.eigenvalues()" << eig_solver.eigenvalues() << std::endl;
	std::cout << "eig_solver.eigenvectors()" << eig_solver.eigenvectors() << std::endl;

	// Indirgeme
	// Eigenvalue matrisi varyans büyük olan sutunlar icin buyuk degerler verir.
	// n boyutlu bir veriyi k boyutlu bir veriye indirgemek icin
	// en buyuk varyansa(eigenvaluelara) sahip k kadar sutun secilir.

	// sirasiyla en buyuk eigenvaluelara sahip k kadar sutunun indekslerini alacagiz.
	// sonra da k sutunlu bir matris olusturup orjinal verimizdeki sutunlari buna atayacagiz.
	Eigen::VectorXd eigvals = eig_solver.eigenvalues();
	// indisler icin basit bir vector olusturalim:
	std::vector< std::pair<int, double> > l;
	std::cout << eigvals.size() << std::endl;
	for (int i = 0; i < eigvals.size(); i++)
	{
		l.push_back(std::pair<int, double>(i, eigvals(i)));
	}
	// sort vector of pairs by second element
	std::sort(l.begin(), l.end(), compare);
	int k = 1;
	// en iyi k elemani al
	for (int i = 0; i<k; i++)
	{
		std::cout << "Indis: " << l[i].first << " Eigenvalue: " << l[i].second << std::endl;
	}	
}

//void pca_test()
//{
//	char filename[] = "dat.txt";
//	std::ifstream ifile(filename);
//	size_t nr, nc;
//	ifile >> nr;
//	ifile >> nc;
//	size_t numEnt = nr * nc;
//	std::vector<double> data(numEnt, 0.0);
//	for (size_t i = 0; i < numEnt; ++i) {
//		ifile >> data[i];
//	}
//	//cout << data.rows
//	std::cout << nr << "X" << nc << endl;
//	Eigen::Map<Eigen::MatrixXd> m(&data[0], nr, nc);
//	//std::cout << m << "\n";
//	PCA p(m);
//	std::cerr << "before decomp\n";
//	p.performDecomposition();
//	auto projected = p.projectedData(0.95);
//	//std::cerr << "projected = " << projected << "\n";
//}
MatrixXd read_data()
{

	//test();
	char filename[] = "data.txt";
	std::ifstream ifile(filename);
	//size_t nr, nc;
	size_t nr, nc;
	ifile >> nr;
	ifile >> nc;
	//cout << nr << "x" << nc;
	size_t numEnt = nr * nc;
	std::vector<double> data(numEnt, 0.0);
	for (size_t i = 0; i < numEnt; ++i) {
		ifile >> data[i];
	}
	//Eigen::Map<Eigen::MatrixXd> mat(&data[0], nr, nc);
	Eigen::Map<Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>> mat(&data[0], nr, nc);

	//cout << mat;
	//cout << mat.rows() << "x" << mat.cols() << endl;
	//cout << mat.block(0, 0, 5, 2) << endl;

	ifile.close();
	return mat;
}

//class PCA {
//public:
//	PCA(Eigen::Map<Eigen::MatrixXd>& m) {
//		dat_ = Eigen::MatrixXd(m.rows(), m.cols());
//		for (size_t i = 0; i < m.rows(); ++i) {
//			for (size_t j = 0; j < m.cols(); ++j) {
//				dat_(i, j) = m(i, j);
//			}
//		}
//	}

void pca_test_example()
{
	Eigen::MatrixXd X;
	X.resize(5, 2);
	X << 1, 5,
		2, 3,
		1, 3,
		5, 1,
		4, 2;
	cout << X << endl;
	//cout << pca.dat_ << "-----------------\n";

	//m(0, 1) = 1;
	//m(0, 0) = 0.5;
	//m(1, 0) = 0.5;
	// Matrisin aritmetik ortalamasi
	std::cout << X.mean() << std::endl;
	// Matris sutunlarinin aritmetik ortalamasi
	std::cout << "m.rowwise().mean()\n" << X.rowwise().mean() << std::endl;

	cout << "X.colwise().mean()\n" << X.colwise().mean() << endl;
	Eigen::MatrixXd aligned = X.rowwise() - X.colwise().mean();
	cout << "alinged\n" << aligned << endl;
	cout << "X\n" << X << endl;
	//Eigen::JacobiSVD<Eigen::MatrixXd> svd(X, ComputeThinV | ComputeThinU );
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(aligned, ComputeFullU | ComputeFullV);
	//Eigen::JacobiSVD<Eigen::MatrixXd> svd(X, Eigen::ComputeFullU | Eigen::ComputeFullV);



	Eigen::MatrixXd cov = (aligned.adjoint() * aligned) / (aligned.rows() - 1);
	//std::cout << "cov" << cov << std::endl;

	MatrixXd v = svd.matrixV();


	//// Eigenvector ayristirmasi
	//Eigen::SelfAdjointEigenSolver<MatrixXd> eig_solver;

	//eig_solver.compute(cov);
	//std::cout << "eig_solver.eigenvalues()\n" << eig_solver.eigenvalues() << std::endl;
	//std::cout << "eig_solver.eigenvectors()\n" << eig_solver.eigenvectors() << std::endl;



	// and here is the question what is the basis matrix and how can i reduce it
	// in my understanding it should be:
	//Eigen::MatrixXd W = svd.matrixV().leftCols(num_components);

	//svd.computeU();
	//svd.computeV();

	cout << "U\n" << svd.matrixU() << endl;
	cout << "V\n" << svd.matrixV() << endl;
	cout << "singular values\n" << svd.singularValues() << endl;
	cout << "covariance\n" << cov << endl;
	//cout << v.leftCols(0) << endl;
	//cout << v.leftCols(1) << endl;
	cout << v.col(0) << endl;
	cout << "eigen values----->\n" << v.col(0).transpose()*cov * v.col(0);
	cout << "eigen values----->\n" << v.col(1).transpose()*cov * v.col(1);


	//MatrixXf m = MatrixXf::Random(3, 2);
	//cout << "Here is the matrix m:" << endl << m << endl;
	//JacobiSVD<MatrixXf> ssvd(m, ComputeThinU | ComputeThinV);
	//cout << "Its singular values are:" << endl << svd.singularValues() << endl;
	//cout << "Its left singular vectors are the columns of the thin U matrix:" << endl << ssvd.matrixU() << endl;
	//cout << "Its right singular vectors are the columns of the thin V matrix:" << endl << ssvd.matrixV() << endl;
	//Vector3f rhs(1, 0, 0);
	//cout << "Now consider this rhs vector:" << endl << rhs << endl;
	//cout << "A least-squares solution of m*x = rhs is:" << endl << ssvd.solve(rhs) << endl;
}
int main(int argc, char *argv[])
{

	//while (!ifile.eof()) {
	//	
	//	//ifile.getline(inputString, 100);
	//	ifile >>

	//	cout << inputString << endl;

	//}

	Eigen::MatrixXd X = read_data();
	
	//X.resize(5, 2);
	//X << 1,5,
	//	2, 3,
	//	1, 3,
	//	5, 1,
	//	4, 2;
	//cout << X << endl;
	cout << X.block(0, 0, 5, 2);
	PCA pca(X);
	cout << "components\n" << pca.components() << endl;
	cout << "explained_variance\n" << pca.explained_variance() << endl;
	///cout << pca.dat_.data() << "," << X.data();
	//Eigen::MatrixXd d;                       // Matrix of doubles.
	Eigen::MatrixXf f = pca.components().cast <float>();   // Matrix of floats.

	cout << f << endl;
	//
	return 0;
}