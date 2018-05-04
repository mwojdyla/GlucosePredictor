#ifndef OUTPUTHANDLER_HPP
#define OUTPUTHANDLER_HPP

#include <QGroupBox>

class QGridLayout;
class QTextEdit;
class QWebEngineView;

class OutputHandler : public QGroupBox
{
    Q_OBJECT

public:
    explicit OutputHandler();
    ~OutputHandler();

    QWebEngineView* chartView_;
    QTextEdit* outputTextView_;
private:
    void configure();

    QGridLayout* mainLayout_;
};

#endif // OUTPUTHANDLER_HPP
