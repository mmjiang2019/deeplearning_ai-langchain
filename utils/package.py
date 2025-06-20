# pip install setuptools
import pkg_resources

installed_packages = pkg_resources.working_set
for dist in installed_packages:
    print(f"{dist.project_name} == {dist.version}")