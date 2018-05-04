#ifndef GLUCOSEPREDICTOR_HPP
#define GLUCOSEPREDICTOR_HPP

#include <QWidget>

class OutputHandler;
class ConfigurationHandler;
class QGridLayout;
class QProcess;

namespace Ui {
class GlucosePredictor;
}

class GlucosePredictor : public QWidget
{
    Q_OBJECT

public:
    explicit GlucosePredictor(QWidget *parent = 0);
    ~GlucosePredictor();

    static int chartNumber;

public slots:
    void onRunButtonClicked() const;
    void onResetButtonClicked() const;
    void onNextChartButtonClicked() const;
    void onPreviousChartButtonClicked() const;
    void onCallback() const;

private:
    bool parametersAreValid() const;
    void runScript() const;
    void resetNetworkParameters() const;
    void displayNextChart() const;
    void displayPreviousChart() const;
    QString makeScriptInvocation() const;
    void concatenateOptionsWithValues(std::string& invocation) const;
    void displayChart() const;

    Ui::GlucosePredictor *ui;
    QGridLayout* mainLayout_;
    OutputHandler* outputBox_;
    ConfigurationHandler* configurationBox_;
    QProcess* predictionAlgorithmTask;
};

#endif // GLUCOSEPREDICTOR_HPP
