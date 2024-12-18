
#include <iostream>
#include <sstream>
#include <string>



void testabc()
{

std::cout << "";
std::cout << "";
std::cout << "";

printf("yosi");
}



#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Appenders/ColorConsoleAppender.h>
//#define LOGI std::stringstream ss;\
//ss

void test()
{}


#/*else
#   define IF_PLOG_(instanceId, severity)   if (!plog::get<instanceId>() || !plog::get<instanceId>()->checkSeverity(severity)) {;} else
#endif

#define IF_PLOG(severity)                IF_PLOG_(PLOG_DEFAULT_INSTANCE_ID, severity)*/

//////////////////////////////////////////////////////////////////////////
// Main logging macros

#define PLOG_(instanceId, severity)      IF_PLOG_(instanceId, severity) (*plog::get<instanceId>()) += plog::Record(severity, PLOG_GET_FUNC(), __LINE__, PLOG_GET_FILE(), PLOG_GET_THIS(), instanceId).ref()
//#define PLOG(severity)                   PLOG_(PLOG_DEFAULT_INSTANCE_ID, severity)


#include <iostream>
#include <sstream>
#include <string>

// Ŭ���� ����
class PrintfStream {
public:

    enum class serverity {
        none = 0,
        debug,
        info,
    };
    std::ostringstream oss;
    serverity maxSeverity;
    PrintfStream(std::string sev = "", serverity val = serverity::none)
        : serverity(sev)
        , maxSeverity(val)
    {};
    std::string serverity;
    // << ������ �����ε�
    template<typename T>
    PrintfStream& operator<<(const T& value) {
        oss << value;
        return *this;
    }

    // << ������ �����ε� (�Ŵ�ǽ������ ����)
    PrintfStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        oss << manip;
        return *this;
    }

    // �Ҹ��ڿ��� printf ȣ��
    ~PrintfStream() {
        printf("[%s]%s", serverity.c_str(), oss.str().c_str());
    }
};

// ��ũ�� ����
#define PRINTF_STRINGIFY PrintfStream("none", PrintfStream::serverity::none)
#define PRINTF_LOGF PrintfStream("fatal")

// Custom class for demonstration
class MyClass {
public:
    int value;
    std::string name;
    MyClass(int val) : value(val) {}

    // Friend function to overload << operatorf5q   
    //friend std::ostream& operator<<(std::ostream& ss, const MyClass& obj);
    //friend std::ostringstream& operator<<(std::ostringstream& ss, const MyClass& obj);
};


// 
// // ����� ���� Ŭ������ ���� << ������ �����ε�
std::ostream& operator<<(std::ostream& os, const MyClass& obj) {
    os << "MyClass(id: " << obj.value << ", name: " << obj.name << ")";
    return os;
}
//std::ostringstream& operator<<(std::ostringstream& ss, const MyClass& obj)
//{
//
//    ss << "MyClass value: " << obj.value;
//    return ss;
//}
////MyClass
//PrintfStream& operator<<(std::stringstream& ss, const MyClass& obj)
//{
//
//}

int main() {

    int x = 42;
    std::string str = "Hello, World!";
    double pi = 3.14159;
    MyClass myObj(42);

    //PrintfStream::serverity::
    // ��ũ�� ��� ����
    PRINTF_STRINGIFY << "The value of x is: " << x << ", str is: " << str << ", and pi is: " << pi << "ojb" << myObj << "\n";
    PRINTF_LOGF << "The value of x is: " << x << ", str is: " << str << ", and pi is: " << pi << "ojb" << myObj  << "\n";

    //static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
    //plog::init(plog::verbose, &consoleAppender);
    //plog::get
    //// Log severity levels are printed in different colors.
    //PLOG_VERBOSE << "This is a VERBOSE message";
    //PLOG_DEBUG << "This is a DEBUG message";
    //PLOG_INFO << "This is an INFO message";
    //PLOG_WARNING << "This is a WARNING message";
    //PLOG_ERROR << "This is an ERROR message";
    //LOGF << "This is a FATAL message";



    //plog::init(plog::debug, "Hello.txt"); // Step2: initialize the logger.

// Step3: write log messages using a special macro. There are several log macros, use the macro you liked the most.

    // Use std::stringstream to capture the output
    std::stringstream ss;
    ss << myObj;

    // Convert the stringstream to a string
    std::string result = ss.str();

    // Output the result to console for demonstration
    std::cout << result << std::endl;

    printf(result.c_str());

    return 0;
}