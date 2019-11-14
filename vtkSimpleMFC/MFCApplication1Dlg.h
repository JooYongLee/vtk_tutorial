
// MFCApplication1Dlg.h : 헤더 파일
//

#pragma once

#include <vtkAutoInit.h>

#define vtkRenderingCore_AUTOINIT 3(vtkRenderingOpenGL2,vtkInteractionStyle, vtkRenderingFreeType)
#define vtkRenderingContext2D_AUTOINIT 1(vtkRenderingContextOpenGL2)

#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkPLYReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkSTLReader.h>

// CMFCApplication1Dlg 대화 상자
class CMFCApplication1Dlg : public CDialogEx
{
// 생성입니다.
public:
	CMFCApplication1Dlg(CWnd* pParent = NULL);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_MFCAPPLICATION1_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.

	vtkNew<vtkRenderWindow> m_vtkRenderWindow;
	vtkNew<vtkRenderer> m_renderer;

	void InitializeVTKWindow(void* hWnd);
	void ResizeVTKWindow();

	void LoadInitData();

	

// 구현입니다.
protected:
	HICON m_hIcon;

	char *filename;

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedLoadButton();
};
