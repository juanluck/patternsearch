#!/usr/bin/env octave
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
1;

x0 = 3.^rand(2,1);
%patternsearch (argv);
patternsearch (x0,0.5);
%pause

