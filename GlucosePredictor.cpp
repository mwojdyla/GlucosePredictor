#include "ConfigurationHandler.hpp"
#include "GlucosePredictor.hpp"
#include "OutputHandler.hpp"
#include "ui_GlucosePredictor.h"

#include <string>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGraphicsView>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QProcess>
#include <QPushButton>
#include <QSpinBox>
#include <QStringList>
#include <QTextEdit>
#include <QThread>
#include <QUrl>
#include <QWebEngineView>
#include <QVBoxLayout>

namespace
{

std::string designateDirFullPath()
{
    QString appDirPath = QDir::currentPath();
    int lastSlashPos = appDirPath.lastIndexOf(QChar('/'));
    QString cuttedDirPath = appDirPath.left(lastSlashPos + 1);
    cuttedDirPath.append("GlucosePredictor/");

    return cuttedDirPath.toUtf8().constData();
}

}

const std::string fullPath = designateDirFullPath();
const std::string scriptName = fullPath + "glucose_prediction.py";
const std::string plotsDirFullPath = "file://" + fullPath + "plots/";
int GlucosePredictor::chartNumber = 1;

GlucosePredictor::GlucosePredictor(QWidget *parent)
: QWidget(parent)
, ui(new Ui::GlucosePredictor)
, mainLayout_(new QGridLayout)
, outputBox_(new OutputHandler)
, configurationBox_(new ConfigurationHandler)
, predictionAlgorithmTask(new QProcess)
{
//    ui->setupUi(this);
    outputBox_->setParent(this);
    configurationBox_->setParent(this);
    mainLayout_->addWidget(outputBox_, 0, 0, 4, 8);
    mainLayout_->addWidget(configurationBox_, 4, 0, 1, 8);
    mainLayout_->setRowStretch(0, 20);
    mainLayout_->setRowStretch(1, 10);

    setLayout(mainLayout_);

    QObject::connect(
        configurationBox_->runButton_, SIGNAL(clicked()),
        this, SLOT(onRunButtonClicked()));
    QObject::connect(
        configurationBox_->resetButton_, SIGNAL(clicked()),
        this, SLOT(onResetButtonClicked()));
    QObject::connect(
        configurationBox_->nextChartButton_, SIGNAL(clicked()),
        this, SLOT(onNextChartButtonClicked()));
    QObject::connect(
        configurationBox_->previousChartButton_, SIGNAL(clicked()),
        this, SLOT(onPreviousChartButtonClicked()));

    QObject::connect(
        predictionAlgorithmTask, SIGNAL(started()),
        configurationBox_, SLOT(onProcessStarted()));
    QObject::connect(
        predictionAlgorithmTask, SIGNAL(finished(int, QProcess::ExitStatus)),
        configurationBox_, SLOT(onProcessFinished(int, QProcess::ExitStatus)));
    QObject::connect(
        configurationBox_, SIGNAL(callback()),
        this, SLOT(onCallback()));
}

GlucosePredictor::~GlucosePredictor()
{
    delete ui;
    delete mainLayout_;
}

void GlucosePredictor::onRunButtonClicked() const
{
    if (parametersAreValid())
    {
        configurationBox_->infoTable_->clear();
        configurationBox_->infoTable_->setText(tr("Computing..."));
        runScript();
    }
}
void GlucosePredictor::onResetButtonClicked() const
{
    configurationBox_->infoTable_->clear();
    resetNetworkParameters();
}
void GlucosePredictor::onNextChartButtonClicked() const
{
    displayNextChart();
}
void GlucosePredictor::onPreviousChartButtonClicked() const
{
    displayPreviousChart();
}

void GlucosePredictor::onCallback() const
{
    QString output(predictionAlgorithmTask->readAllStandardOutput());
    predictionAlgorithmTask->close();

    outputBox_->outputTextView_->clear();
    outputBox_->outputTextView_->setText(output);
    displayChart();

    if (configurationBox_->folds_->value() > chartNumber)
    {
        configurationBox_->nextChartButton_->setEnabled(true);
    }

    configurationBox_->infoTable_->append(tr("Finished"));
}

bool GlucosePredictor::parametersAreValid() const
{
    bool areValid = true;
    configurationBox_->infoTable_->clear();
    configurationBox_->infoTable_->setTextColor(Qt::red);

    configurationBox_->infoTable_->setText(tr("Some parameters are not valid:"));
    if (configurationBox_->samples_->value() == 0)
    {
        areValid = false;
        configurationBox_->infoTable_->append(
            tr("*Samples' number must be different than zero!"));
    }
    if (configurationBox_->folds_->value() == 0)
    {
        areValid = false;
        configurationBox_->infoTable_->append(
            tr("*Folds' number must be different than zero!"));
    }
    if (configurationBox_->rows_->value() == 0)
    {
        areValid = false;
        configurationBox_->infoTable_->append(
            tr("*Rows' number must be different than zero!"));
    }
    if (configurationBox_->columns_->value() == 0)
    {
        areValid = false;
        configurationBox_->infoTable_->append(
            tr("*Columns' number must be different than zero!"));
    }
    if (configurationBox_->epochs_->value() == 0)
    {
        areValid = false;
        configurationBox_->infoTable_->append(
            tr("*Epochs' number must be different than zero!"));
    }
    if (configurationBox_->learningRate_->value() == 0.0)
    {
        areValid = false;
        configurationBox_->infoTable_->append(
            tr("*Learning rate should be bigger than zero!"));
    }
    configurationBox_->infoTable_->setTextColor(Qt::black);

    return areValid;
}

void GlucosePredictor::runScript() const
{
    QString invocation = makeScriptInvocation();
    predictionAlgorithmTask->setProcessChannelMode(QProcess::MergedChannels);
    predictionAlgorithmTask->start(invocation);
}

QString GlucosePredictor::makeScriptInvocation() const
{
    std::string language = "python3 ";
    std::string scriptInvocation = language + scriptName;

    concatenateOptionsWithValues(scriptInvocation);

    return tr(scriptInvocation.c_str());
}

void GlucosePredictor::concatenateOptionsWithValues(std::string& invocation) const
{
    invocation += " -D " + std::to_string(configurationBox_->samples_->value());
    invocation += " -F " + std::to_string(configurationBox_->folds_->value());
    invocation += " -R " + std::to_string(configurationBox_->rows_->value());
    invocation += " -C " + std::to_string(configurationBox_->columns_->value());
    invocation += " -E " + std::to_string(configurationBox_->epochs_->value());
    std::string rate = std::to_string(configurationBox_->learningRate_->value());
    const size_t dotIndex = rate.find(",");
    rate.replace(dotIndex, std::string(",").length(), ".");
    invocation += " -L " + rate;
}

void GlucosePredictor::displayChart() const
{
    std::string chartPath = plotsDirFullPath + "plot" + std::to_string(chartNumber) + ".html";
    outputBox_->chartView_->load(QUrl(tr(chartPath.c_str())));
}

void GlucosePredictor::resetNetworkParameters() const
{
    configurationBox_->samples_->setValue(0);
    configurationBox_->folds_->setValue(0);
    configurationBox_->rows_->setValue(0);
    configurationBox_->columns_->setValue(0);
    configurationBox_->epochs_->setValue(0);
    configurationBox_->learningRate_->setValue(0.0);
    configurationBox_->infoTable_->clear();
}

void GlucosePredictor::displayNextChart() const
{
    chartNumber++;
    displayChart();

    if (!configurationBox_->previousChartButton_->isEnabled())
    {
        configurationBox_->previousChartButton_->setEnabled(true);
    }

    if (configurationBox_->folds_->value() == chartNumber)
    {
        configurationBox_->nextChartButton_->setDisabled(true);
    }
}

void GlucosePredictor::displayPreviousChart() const
{
    chartNumber--;
    displayChart();

    if (!configurationBox_->nextChartButton_->isEnabled())
    {
        configurationBox_->nextChartButton_->setEnabled(true);
    }

    if (chartNumber == 1)
    {
        configurationBox_->previousChartButton_->setDisabled(true);
    }
}
