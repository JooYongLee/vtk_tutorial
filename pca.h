#ifndef PCA_H_
#define PCA_H_
#include <Eigen/Dense>
// 파이썬 sklearn 네이밍과 유사하게 작성함
class PCA
{
public:	
	PCA(Eigen::MatrixXd &m);

	~PCA() = default;

	bool compute();

	Eigen::MatrixXd& components() { return _components; } 

	Eigen::VectorXd& explained_variance()  { return _variances; }

private:
	// input matrix
	Eigen::MatrixXd _dat;
	// transform matrix
	Eigen::MatrixXd _proj;
	// convariance matrix
	Eigen::MatrixXd _cov;
	// eigen values, 
	Eigen::VectorXd _variances;
	// eigen vector, components
	Eigen::MatrixXd _components;
};

#endif //PCA_H_
