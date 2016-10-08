"""
BenchExec is a framework for reliable benchmarking.
This file is part of BenchExec.

Copyright (C) 2007-2015  Dirk Beyer
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import benchexec.util as util
import benchexec.tools.template
import benchexec.result as result

class Tool(benchexec.tools.template.BaseTool):

    def executable(self):
        return util.find_executable('satabs')


    def version(self, executable):
        return self._version_from_tool(executable)


    def name(self):
        return 'SatAbs'


    def determine_result(self, returncode, returnsignal, output, isTimeout):
        output = '\n'.join(output)
        if "VERIFICATION SUCCESSFUL" in output:
            assert returncode == 0
            status = result.RESULT_TRUE_PROP
        elif "VERIFICATION FAILED" in output:
            assert returncode == 10
            status = result.RESULT_FALSE_REACH
        elif returnsignal == 9:
            status = "TIMEOUT"
        elif returnsignal == 6:
            if "Assertion `!counterexample.steps.empty()' failed" in output:
                status = 'COUNTEREXAMPLE FAILED' # TODO: other status?
            else:
                status = "OUT OF MEMORY"
        elif returncode == 1 and "PARSING ERROR" in output:
            status = "PARSING ERROR"
        else:
            status = "FAILURE"
        return status
