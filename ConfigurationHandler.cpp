#include <string>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QTextEdit>
#include <QVBoxLayout>

#include "ConfigurationHandler.hpp"

ConfigurationHandler::ConfigurationHandler(ObjectFactoryPtr factory)
: QGroupBox(tr("Configuration view"))
, factory_(factory)
, mainLayout_(new QGridLayout)
{
    configure();
}

ConfigurationHandler::~ConfigurationHandler()
{
    delete buttonsBox_;
    delete parametersBox_;
    delete runButton_;
    delete resetButton_;
    delete nextChartButton_;
    delete previousChartButton_;
    delete samples_;
    delete folds_;
    delete rows_;
    delete columns_;
    delete epochs_;
    delete learningRate_;
    delete infoTable_;
}

void ConfigurationHandler::configure()
{
    createButtons();
    createParametersBox();
    infoTable_ = new QTextEdit;

    infoTable_->setFontPointSize(12.0);
    infoTable_->setReadOnly(true);

    mainLayout_->addWidget(buttonsBox_, 0, 0, 1, 1);
    mainLayout_->addWidget(parametersBox_, 0, 1, 1, 2);
    mainLayout_->addWidget(infoTable_, 0, 3, 1, 2);

    setLayout(mainLayout_);
}

void ConfigurationHandler::createButtons()
{
    buttonsBox_ = new QGroupBox;
    runButton_ = new QPushButton(tr("Run"));
    resetButton_ = new QPushButton(tr("Reset"));
    nextChartButton_ = new QPushButton(tr("Next chart"));
    previousChartButton_ = new QPushButton(tr("Previous chart"));

    nextChartButton_->setDisabled(true);
    previousChartButton_->setDisabled(true);

    QVBoxLayout* const buttonsBoxLayout = new QVBoxLayout;
    setButtonsLayout(buttonsBoxLayout);
}

void ConfigurationHandler::setButtonsLayout(QVBoxLayout* const layout) const
{
    layout->addWidget(runButton_);
    layout->addWidget(resetButton_);
    layout->addWidget(nextChartButton_);
    layout->addWidget(previousChartButton_);

    buttonsBox_->setLayout(layout);
}

void ConfigurationHandler::createParametersBox()
{
    parametersBox_ = new QGroupBox;
    samples_ = new QSpinBox;
    folds_ = new QSpinBox;
    rows_ = new QSpinBox;
    columns_ = new QSpinBox;
    epochs_ = new QSpinBox;
    learningRate_ = new QDoubleSpinBox;

    samples_->setMaximum(100000);
    folds_->setMaximum(1000);
    rows_->setMaximum(1000);
    columns_->setMaximum(1000);
    epochs_->setMaximum(100000);
    learningRate_->setMaximum(3.0);
    learningRate_->setSingleStep(0.1);

    samples_->setValue(1000);
    folds_->setValue(20);
    rows_->setValue(4);
    columns_->setValue(5);
    epochs_->setValue(100);
    learningRate_->setValue(2.0);

    QFormLayout* const parametersBoxLayout = new QFormLayout;
    setParametersLayout(parametersBoxLayout);
}

void ConfigurationHandler::setParametersLayout(QFormLayout* const layout) const
{
    layout->addRow(new QLabel(tr("Samples")), samples_);
    layout->addRow(new QLabel(tr("Folds")), folds_);
    layout->addRow(new QLabel(tr("Rows")), rows_);
    layout->addRow(new QLabel(tr("Columns")), columns_);
    layout->addRow(new QLabel(tr("Epochs")), epochs_);
    layout->addRow(new QLabel(tr("Learning rate")), learningRate_);

    parametersBox_->setLayout(layout);
}

void ConfigurationHandler::onProcessStarted() const
{
    blockButtons();
}

void ConfigurationHandler::onProcessFinished(const int exitCode,
    const QProcess::ExitStatus status) const
{
    handleExitCodeAndStatus(exitCode, status);
    unblockButtons();
    emit callback();
}

void ConfigurationHandler::handleExitCodeAndStatus(const int exitCode,
    const QProcess::ExitStatus status) const
{
    if (exitCode < 0 || status != QProcess::NormalExit)
    {
        const std::string exitCodeToStr = std::to_string(exitCode);
        const std::string statusToStr = status == QProcess::CrashExit ? "Crash" : "Normal";
        const std::string message =
            "Code of the completed process is " + exitCodeToStr + " with status " + statusToStr;

        infoTable_->clear();
        infoTable_->setTextColor(Qt::red);
        infoTable_->setText(tr(message.c_str()));
        infoTable_->setTextColor(Qt::black);
    }
}

void ConfigurationHandler::blockButtons() const
{
    if (runButton_->isEnabled())
    {
        runButton_->setDisabled(true);
    }
    if (resetButton_->isEnabled())
    {
        resetButton_->setDisabled(true);
    }
    if (nextChartButton_->isEnabled())
    {
        nextChartButton_->setDisabled(true);
    }
    if (previousChartButton_->isEnabled())
    {
        previousChartButton_->setDisabled(true);
    }
}

void ConfigurationHandler::unblockButtons() const
{
    runButton_->setEnabled(true);
    resetButton_->setEnabled(true);
}

