#ifndef CONFIGURATIONHANDLER_HPP
#define CONFIGURATIONHANDLER_HPP

#include <QGroupBox>
#include <QProcess>

class QDoubleSpinBox;
class QFormLayout;
class QGridLayout;
class QPushButton;
class QSpinBox;
class QTextEdit;
class QVBoxLayout;

class ConfigurationHandler : public QGroupBox
{
    Q_OBJECT

public:
    explicit ConfigurationHandler();
    ~ConfigurationHandler();

    QGroupBox* buttonsBox_;
    QGroupBox* parametersBox_;
    QPushButton* runButton_;
    QPushButton* resetButton_;
    QPushButton* nextChartButton_;
    QPushButton* previousChartButton_;
    QSpinBox* samples_;
    QSpinBox* folds_;
    QSpinBox* rows_;
    QSpinBox* columns_;
    QSpinBox* epochs_;
    QDoubleSpinBox* learningRate_;
    QTextEdit* infoTable_;

public slots:
    void onProcessStarted() const;
    void onProcessFinished(const int exitCode,
        const QProcess::ExitStatus status) const;

signals:
    void callback() const;

private:
    void handleExitCodeAndStatus(const int exitCode,
        const QProcess::ExitStatus status) const;
    void blockButtons() const;
    void unblockButtons() const;
    void configure();
    void createButtons();
    void setButtonsLayout(QVBoxLayout* const layout) const;
    void createParametersBox();
    void setParametersLayout(QFormLayout* const layout) const;

    QGridLayout* const mainLayout_;
};

#endif // CONFIGURATIONHANDLER_HPP
