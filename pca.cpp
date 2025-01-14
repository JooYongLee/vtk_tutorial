#include "pca.h"
using namespace Eigen;

PCA::PCA(Eigen::MatrixXd &m)
{
	_dat = m;
	this->compute();
};










bool PCA::compute()
{
	Eigen::MatrixXd aligned = _dat.rowwise() - _dat.colwise().mean();

	Eigen::JacobiSVD<Eigen::MatrixXd> svd(aligned, Eigen::ComputeFullV);

	Eigen::MatrixXd cov = (aligned.adjoint() * aligned) / (aligned.rows() - 1);

	this->_cov = cov;

	Eigen::MatrixXd v = svd.matrixV();

	long long m = _dat.cols();

	_components = svd.matrixV().transpose();
	_variances.resize(m);
	for (int i = 0; i < m; i++)
	{
		_variances(i) = _components.row(i) * _cov * _components.row(i).transpose();
	}

	return true;
}