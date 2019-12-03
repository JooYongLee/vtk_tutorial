#include <vtkFloatArray.h>
#include <vtkIdTypeArray.h>
#include <vtkSmartPointer.h>
#include <iostream>

using namespace std;

void test_vtkarray_access()
{
	
	vtkSmartPointer<vtkFloatArray> data =\
		vtkSmartPointer<vtkFloatArray>::New();

	//
	const int n = 5;
	const int d = 3;
	
	data->SetNumberOfComponents(d);
	data->SetNumberOfTuples(n);
	float ptr[n*d] = { 0, };
	for (int i = 0; i < n*d; i++)
	{
		ptr[i] = i * 0.1f;
		//cout << ptr[i] << endl;
		//data->GetValue()
	}

	memcpy(data->GetVoidPointer(0), ptr, sizeof(float)*n*d);
	cout << "copy complete:" << data->GetVoidPointer(0) <<endl;
	//
	//double p[3];
	for (int i = 0; i < n; i++)
	{
		//float *p = data->GetTuple(i);
		//data->GetTuple(i, p);
		double *p = data->GetTuple(i);
		for (int k = 0; k < d; k++)
		{
			cout << p[k] << ",";
		}
		cout << endl;
		
	}
	//data->Delete();

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	vtkSmartPointer<vtkIdTypeArray> idata = \
		vtkSmartPointer<vtkIdTypeArray>::New();
	// make sure that set number of Components first prior to Tuples
	idata->SetNumberOfComponents(d);
	idata->SetNumberOfTuples(n);
	
	vtkIdType ptr_long[n*d] = { 0, };
	for (int i = 0; i < n*d; i++)
	{
		ptr_long[i] = i;
		//cout << ptr_long[i] << endl;
	}
	memcpy(idata->GetVoidPointer(0), ptr_long, sizeof(long long)*n*d);

	long long *pointr = static_cast<long long*>(idata->GetVoidPointer(0));

	for (int i = 0; i < n; i++)
	{
		for (int k = 0; k < d; k++)
		{
			//cout << p[k] << ",";
			cout << *(pointr++) << ",";			
		}
		cout << endl;
	}
}
