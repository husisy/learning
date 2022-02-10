lappend auto_path [file normalize user_package]
package require HelloWorld 1.0
puts [HelloWorld::MyProcedure]
puts [HelloWorld MyProcedure]
