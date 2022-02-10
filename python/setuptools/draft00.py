from pkg_resources import parse_version

parse_version('1.9.a.dev')==parse_version('1.9a0dev')
parse_version('2.1-rc2') < parse_version('2.1')
parse_version('0.6a9dev-r41475') < parse_version('0.6a9')
