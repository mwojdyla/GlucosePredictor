#ifndef OBJECTFACTORY_HPP
#define OBJECTFACTORY_HPP

#include <memory>

class QWidget;
class QLayout;

enum class WidgetType
{
    DoubleSpinBox,
    GroupBox,
    PushButton,
    SpinBox,
    TextEdit,
    WebEngineView
};

enum class LayoutType
{
    FormLayout,
    GridLayout,
    VBoxLayout
};

class ObjectFactory
{
public:
    explicit ObjectFactory() = default;

    std::shared_ptr<QWidget> makeWidget(const WidgetType type) const;
    std::shared_ptr<QLayout> makeLayout(const LayoutType type) const;
};

using ObjectFactoryPtr = const ObjectFactory* const;

#endif // OBJECTFACTORY_HPP
