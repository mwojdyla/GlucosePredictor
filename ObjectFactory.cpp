#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QPushButton>
#include <QSpinBox>
#include <QTextEdit>
#include <QWebEngineView>

#include <QFormLayout>
#include <QGridLayout>
#include <QVBoxLayout>

#include "ObjectFactory.hpp"

std::shared_ptr<QWidget> ObjectFactory::makeWidget(const WidgetType type) const
{
    switch (type)
    {
        case WidgetType::DoubleSpinBox:
            return std::make_shared<QDoubleSpinBox>(nullptr);
        case WidgetType::GroupBox:
            return std::make_shared<QGroupBox>(nullptr);
        case WidgetType::PushButton:
            return std::make_shared<QPushButton>(nullptr);
        case WidgetType::SpinBox:
            return std::make_shared<QSpinBox>(nullptr);
        case WidgetType::TextEdit:
            return std::make_shared<QTextEdit>(nullptr);
        case WidgetType::WebEngineView:
            return std::make_shared<QWebEngineView>(nullptr);
    }

    return nullptr;
}

std::shared_ptr<QLayout> ObjectFactory::makeLayout(const LayoutType type) const
{
    switch (type)
    {
        case LayoutType::FormLayout:
            return std::make_shared<QFormLayout>(nullptr);
        case LayoutType::GridLayout:
            return std::make_shared<QGridLayout>(nullptr);
        case LayoutType::VBoxLayout:
            return std::make_shared<QVBoxLayout>(nullptr);
    }

    return nullptr;
}


