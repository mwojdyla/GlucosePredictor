#include <QGridLayout>
#include <QTextEdit>
#include <QWebEngineView>

#include "OutputHandler.hpp"

OutputHandler::OutputHandler(ObjectFactoryPtr factory)
: QGroupBox(tr("Output view"))
, chartView_(new QWebEngineView)
, outputTextView_(new QTextEdit)
, factory_(factory)
, mainLayout_(new QGridLayout)
{
    configure();
}

OutputHandler::~OutputHandler()
{
    delete chartView_;
    delete outputTextView_;
    delete mainLayout_;
}

void OutputHandler::configure()
{
    outputTextView_->setFontPointSize(12.0);
    outputTextView_->setReadOnly(true);

    mainLayout_->addWidget(chartView_, 0, 0, 3, 6);
    mainLayout_->addWidget(outputTextView_, 0, 6, 3, 2);
    mainLayout_->setColumnStretch(0, 20);
    mainLayout_->setColumnStretch(6, 6);

    setLayout(mainLayout_);
}
