# https://www.tutorialspoint.com/tcl-tk/tcl_packages.htm
namespace eval ::HelloWorld {
    # Export MyProcedure
    namespace export MyProcedure

    namespace ensemble create

    set version 1.0
    set MyDescription "HelloWorld233"

    # Variable for the path of the script
    variable home [file join [pwd] [file dirname [info script]]]
}

# Definition of the procedure MyProcedure
proc ::HelloWorld::MyProcedure {} {
    puts $HelloWorld::MyDescription
}

package provide HelloWorld $HelloWorld::version
package require Tcl 8.6
