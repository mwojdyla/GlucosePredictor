#include "GlucosePredictor.hpp"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
//    QDir::setCurrent();
    GlucosePredictor w;
    w.showMaximized();

    return a.exec();
}
