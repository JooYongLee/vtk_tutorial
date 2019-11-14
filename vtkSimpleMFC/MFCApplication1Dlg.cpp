
// MFCApplication1Dlg.cpp : ���� ����
//

#include "stdafx.h"
#include "MFCApplication1.h"
#include "MFCApplication1Dlg.h"
#include "afxdialogex.h"

#include <vtkFloatArray.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// ���� ���α׷� ������ ���Ǵ� CAboutDlg ��ȭ �����Դϴ�.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// ��ȭ ���� �������Դϴ�.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV �����Դϴ�.

// �����Դϴ�.
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


// CMFCApplication1Dlg ��ȭ ����



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


// CMFCApplication1Dlg �޽��� ó����

BOOL CMFCApplication1Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// �ý��� �޴��� "����..." �޴� �׸��� �߰��մϴ�.

	// IDM_ABOUTBOX�� �ý��� ��� ������ �־�� �մϴ�.
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

	// �� ��ȭ ������ �������� �����մϴ�.  ���� ���α׷��� �� â�� ��ȭ ���ڰ� �ƴ� ��쿡��
	//  �����ӿ�ũ�� �� �۾��� �ڵ����� �����մϴ�.
	SetIcon(m_hIcon, TRUE);			// ū �������� �����մϴ�.
	SetIcon(m_hIcon, FALSE);		// ���� �������� �����մϴ�.

	// TODO: ���⿡ �߰� �ʱ�ȭ �۾��� �߰��մϴ�.
	InitializeVTKWindow(GetDlgItem(IDC_PC_CHART)->GetSafeHwnd());
	ResizeVTKWindow();

	
	return TRUE;  // ��Ŀ���� ��Ʈ�ѿ� �������� ������ TRUE�� ��ȯ�մϴ�.
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

// ��ȭ ���ڿ� �ּ�ȭ ���߸� �߰��� ��� �������� �׸�����
//  �Ʒ� �ڵ尡 �ʿ��մϴ�.  ����/�� ���� ����ϴ� MFC ���� ���α׷��� ��쿡��
//  �����ӿ�ũ���� �� �۾��� �ڵ����� �����մϴ�.

void CMFCApplication1Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // �׸��⸦ ���� ����̽� ���ؽ�Ʈ�Դϴ�.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Ŭ���̾�Ʈ �簢������ �������� ����� ����ϴ�.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// �������� �׸��ϴ�.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// ����ڰ� �ּ�ȭ�� â�� ���� ���ȿ� Ŀ���� ǥ�õǵ��� �ý��ۿ���
//  �� �Լ��� ȣ���մϴ�.
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
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.
}


void CMFCApplication1Dlg::OnBnClickedLoadButton()
{
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.
	// TODO: Add your control notification handler code here

	static TCHAR BASED_CODE szFilter[] = _T("stl ����(*.stl) | *.stl; |�������(*.*)|*.*||");

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
