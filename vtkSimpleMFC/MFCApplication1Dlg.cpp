
// MFCApplication1Dlg.cpp : 구현 파일
//

#include "stdafx.h"
#include "MFCApplication1.h"
#include "MFCApplication1Dlg.h"
#include "afxdialogex.h"

#include <vtkFloatArray.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CMFCApplication1Dlg 대화 상자



CMFCApplication1Dlg::CMFCApplication1Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_MFCAPPLICATION1_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CMFCApplication1Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CMFCApplication1Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()	
	ON_BN_CLICKED(IDC_LOAD_BUTTON, &CMFCApplication1Dlg::OnBnClickedLoadButton)
END_MESSAGE_MAP()


// CMFCApplication1Dlg 메시지 처리기

BOOL CMFCApplication1Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.
	InitializeVTKWindow(GetDlgItem(IDC_PC_CHART)->GetSafeHwnd());
	ResizeVTKWindow();

	
	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void CMFCApplication1Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 응용 프로그램의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CMFCApplication1Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CMFCApplication1Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CMFCApplication1Dlg::InitializeVTKWindow(void* hWnd)
{
	vtkNew<vtkRenderWindowInteractor> interactor;

	interactor->SetInteractorStyle(vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New());

	//vtkNew<vtkRenderer> renderer;
	//renderer->SetBackground(0.1, 0.2, 0.3);

	m_vtkRenderWindow->SetParentId(hWnd);
	m_vtkRenderWindow->SetInteractor(interactor);
	m_vtkRenderWindow->AddRenderer(m_renderer);
	m_vtkRenderWindow->Render();

	OutputDebugStringA("==================================\n==================================\n");
}

void CMFCApplication1Dlg::ResizeVTKWindow()
{
	CRect rc;
	GetDlgItem(IDC_PC_CHART)->GetClientRect(rc);
	m_vtkRenderWindow->SetSize(rc.Width(), rc.Height());
}


void CMFCApplication1Dlg::LoadInitData()
{
	vtkNew<vtkSTLReader> pPLYReader;
	printf("loading..........%s\n", this->filename);
	pPLYReader->SetFileName(this->filename);
	pPLYReader->Update();

	vtkSmartPointer<vtkPolyData> polyData = pPLYReader->GetOutput();
	vtkFloatArray *float_array = vtkFloatArray::SafeDownCast(polyData->GetPoints()->GetData());
	vtkSmartPointer<vtkFloatArray> distances =
		vtkSmartPointer<vtkFloatArray>::New();
	distances->SetName("Distances");
	distances->SetNumberOfComponents(3);
	distances->SetNumberOfTuples(5);


	vtkNew<vtkPolyDataMapper> mapper;
	mapper->SetInputConnection(pPLYReader->GetOutputPort());

	

	vtkNew<vtkActor> actor;
	actor->SetMapper(mapper);
	
	
	m_renderer->AddActor(actor);
	m_renderer->SetBackground(.1, .2, .3);
	m_renderer->ResetCamera();

	//m_vtkRenderWindow->AddRenderer(renderer);
	m_vtkRenderWindow->Render();

	//renderer->ResetCamera();
	m_vtkRenderWindow->Render();
}

void CMFCApplication1Dlg::OnBnClickedButton1()
{
	OutputDebugStringA("==================================\n==================================\n");
	OutputDebugString(L"dsfsdfdsf");
	printf("Sdfsdfsdf");
	this->LoadInitData();
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
}


void CMFCApplication1Dlg::OnBnClickedLoadButton()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	// TODO: Add your control notification handler code here

	static TCHAR BASED_CODE szFilter[] = _T("stl 파일(*.stl) | *.stl; |모든파일(*.*)|*.*||");

	CFileDialog dlg(TRUE, _T("*.stl"), _T("scan"), OFN_HIDEREADONLY, szFilter);

	if (IDOK == dlg.DoModal())

	{

		CString pathName = dlg.GetPathName();
		//pathName.std
		int length = pathName.GetLength();
		char st[100] = "dsfsdfdsf";
		char* ss = LPSTR(LPCTSTR(pathName));
		pathName.GetString();
		printf("%s\n", st);
		strcpy_s(st, CT2A(pathName));
		printf("%s\n", st);
		this->filename = st;


		this->LoadInitData();
	
		
		


		MessageBox(pathName);

	}
}
