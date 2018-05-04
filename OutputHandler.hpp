#ifndef OUTPUTHANDLER_HPP
#define OUTPUTHANDLER_HPP

#include <QGroupBox>

#include "ObjectFactory.hpp"

class QGridLayout;
class QTextEdit;
class QWebEngineView;

class OutputHandler : public QGroupBox
{
    Q_OBJECT

public:
    explicit OutputHandler(ObjectFactoryPtr factory);
    ~OutputHandler();

    QWebEngineView* chartView_;
    QTextEdit* outputTextView_;
private:
    void configure();

    ObjectFactoryPtr factory_;
    QGridLayout* mainLayout_;
};

#endif // OUTPUTHANDLER_HPP
