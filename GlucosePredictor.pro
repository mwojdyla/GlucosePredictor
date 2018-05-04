#-------------------------------------------------
#
# Project created by QtCreator 2018-04-11T23:08:27
#
#-------------------------------------------------

QT       += core gui
QT       += charts
QT       += webenginewidgets
CONFIG   += c++14
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = GlucosePredictor
TEMPLATE = app


SOURCES += main.cpp\
        GlucosePredictor.cpp \
    ConfigurationHandler.cpp \
    OutputHandler.cpp \
    ObjectFactory.cpp

HEADERS  += GlucosePredictor.hpp \
    ConfigurationHandler.hpp \
    OutputHandler.hpp \
    ObjectFactory.hpp

FORMS    += GlucosePredictor.ui
