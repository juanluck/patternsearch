#!/usr/bin/octave
## Copyright (C) 2020 juanlu
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} patternsearch (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: juanlu <juanlu@verne>
## Created: 2020-09-14


args = argv();
x0 = strread (args{1,1},"%f");
alpha0 = strread (args{2,1},"%f");
objectivefunction = args{3,1};
basis = args{4,1};
order = args{5,1};
tauplus = strread (args{6,1},"%f");
tauminus = strread (args{7,1},"%f");
%x0 = 3.*rand(2,1);
%patternsearch ();
patternsearch(x0,alpha0,objectivefunction,basis,order,tauplus,tauminus);
%pause

