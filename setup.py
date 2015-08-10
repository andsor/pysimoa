#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for simoa.

    This file was generated with PyScaffold 2.2, a tool that easily
    puts up a scaffold for your new Python project. Learn more under:
    http://pyscaffold.readthedocs.org/
"""

import inspect
import os
import sys
from distutils.cmd import Command
from distutils.filelist import FileList

import setuptools
from setuptools import setup

# For Python 2/3 compatibility, pity we can't use six.moves here
try:  # try Python 3 imports first
    import configparser
except ImportError:  # then fall back to Python 2
    import ConfigParser as configparser

__location__ = os.path.join(os.getcwd(), os.path.dirname(
    inspect.getfile(inspect.currentframe())))

# determine root package and package path if namespace package is used
pyscaffold_version = "2.2"
package = "simoa"
namespace = []
root_pkg = namespace[0] if namespace else package
if namespace:
    pkg_path = os.path.join(*namespace[-1].split('.') + [package])
else:
    pkg_path = package


def version2str(version):
    if version.exact or not version.distance > 0:
        return version.format_with('{tag}')
    else:
        distance = version.distance
        version = str(version.tag)
        if '.dev' in version:
            version, tail = version.rsplit('.dev', 1)
            assert tail == '0', 'own dev numbers are unsupported'
        return '{}.post0.dev{}'.format(version, distance)


def local_version2str(version):
    if version.exact:
        if version.dirty:
            return version.format_with('+dirty')
        else:
            return ''
    else:
        if version.dirty:
            return version.format_with('+{node}.dirty')
        else:
            return version.format_with('+{node}')


class ObjKeeper(type):
    instances = {}

    def __init__(cls, name, bases, dct):
        cls.instances[cls] = []

    def __call__(cls, *args, **kwargs):
        cls.instances[cls].append(super(ObjKeeper, cls).__call__(*args,
                                                                 **kwargs))
        return cls.instances[cls][-1]


def capture_objs(cls):
    from six import add_metaclass
    module = inspect.getmodule(cls)
    name = cls.__name__
    keeper_class = add_metaclass(ObjKeeper)(cls)
    setattr(module, name, keeper_class)
    cls = getattr(module, name)
    return keeper_class.instances[cls]


def get_install_requirements(path):
    with open(os.path.join(__location__, path)) as fh:
        content = fh.read()
    return [req for req in content.splitlines() if req != '']


def read(fname):
    with open(os.path.join(__location__, fname)) as fh:
        content = fh.read()
    return content


def str2bool(val):
    return val.lower() in ("yes", "true")


def get_items(parser, section):
    try:
        items = parser.items(section)
    except configparser.NoSectionError:
        return []
    return items


def prepare_console_scripts(dct):
    return ['{cmd} = {func}'.format(cmd=k, func=v) for k, v in dct.items()]


def prepare_extras_require(dct):
    return {k: [r.strip() for r in v.split(',')] for k, v in dct.items()}


def prepare_data_files(dct):
    def get_files(pattern):
        filelist = FileList()
        if '**' in pattern:
            pattern = pattern.replace('**', '*')
            anchor = False
        else:
            anchor = True
        filelist.include_pattern(pattern, anchor)
        return filelist.files

    return [(k, [f for p in v.split(',') for f in get_files(p.strip())])
            for k, v in dct.items()]


def read_setup_cfg():
    config = configparser.SafeConfigParser(allow_no_value=True)
    config_file = os.path.join(__location__, 'setup.cfg')
    with open(config_file, 'r') as f:
        config.readfp(f)
    metadata = dict(config.items('metadata'))
    classifiers = metadata.get('classifiers', '')
    metadata['classifiers'] = [item.strip() for item in classifiers.split(',')]
    console_scripts = dict(get_items(config, 'console_scripts'))
    console_scripts = prepare_console_scripts(console_scripts)
    extras_require = dict(get_items(config, 'extras_require'))
    extras_require = prepare_extras_require(extras_require)
    data_files = dict(get_items(config, 'data_files'))
    data_files = prepare_data_files(data_files)
    package_data = metadata.get('package_data', '')
    package_data = [item.strip() for item in package_data.split(',') if item]
    metadata['package_data'] = package_data
    return metadata, console_scripts, extras_require, data_files


def build_cmd_docs():
    try:
        from sphinx.setup_command import BuildDoc
    except ImportError:
        class NoSphinx(Command):
            user_options = []

            def initialize_options(self):
                raise RuntimeError("Sphinx documentation is not installed, "
                                   "run: pip install sphinx")

        return NoSphinx

    class cmd_docs(BuildDoc):

        def set_version(self):
            from setuptools_scm import get_version
            self.release = get_version()
            self.version = self.release.split('-', 1)[0]

        def run(self):
            self.set_version()
            if self.builder == "doctest":
                import sphinx.ext.doctest as doctest
                # Capture the DocTestBuilder class in order to return the total
                # number of failures when exiting
                ref = capture_objs(doctest.DocTestBuilder)
                BuildDoc.run(self)
                errno = ref[-1].total_failures
                sys.exit(errno)
            else:
                BuildDoc.run(self)

    return cmd_docs


# Assemble everything and call setup(...)
def setup_package():
    docs_path = os.path.join(__location__, "docs")
    docs_build_path = os.path.join(docs_path, "_build")
    needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
    pytest_runner = ['pytest-runner'] if needs_pytest else []
    install_reqs = get_install_requirements("requirements.txt")
    metadata, console_scripts, extras_require, data_files = read_setup_cfg()

    command_options = {
        'docs': {'project': ('setup.py', 'pysimoa'),
                 'build_dir': ('setup.py', docs_build_path),
                 'config_dir': ('setup.py', docs_path),
                 'source_dir': ('setup.py', docs_path)},
        'doctest': {'project': ('setup.py', package),
                    'build_dir': ('setup.py', docs_build_path),
                    'config_dir': ('setup.py', docs_path),
                    'source_dir': ('setup.py', docs_path),
                    'builder': ('setup.py', 'doctest')}
    }

    setup(name=package,
          url=metadata['url'],
          description=metadata['description'],
          author=metadata['author'],
          author_email=metadata['author_email'],
          license=metadata['license'],
          long_description=read('README.rst'),
          classifiers=metadata['classifiers'],
          test_suite='tests',
          packages=setuptools.find_packages(exclude=['tests', 'tests.*']),
          namespace_packages=namespace,
          install_requires=install_reqs,
          setup_requires=['six', 'setuptools_scm'] + pytest_runner,
          extras_require=extras_require,
          cmdclass={'docs': build_cmd_docs(), 'doctest': build_cmd_docs()},
          tests_require=['pytest-cov', 'pytest'],
          package_data={package: metadata['package_data']},
          data_files=data_files,
          command_options=command_options,
          entry_points={'console_scripts': console_scripts},
          use_scm_version={'version_scheme': version2str,
                           'local_scheme': local_version2str},
          zip_safe=False)  # do not zip egg file after setup.py install


if __name__ == "__main__":
    setup_package()
