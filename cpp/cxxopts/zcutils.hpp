#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cassert>

std::string argparse_bool(int argc, char *argv[], std::string key, bool default_value=false);
std::string argparse_string(int argc, char *argv[], std::string key, std::string default_value="");
//TODO --help
//TODO Options option(argc, argv)
//TODO help info
//TODO print_parse_info
