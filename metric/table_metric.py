# Copyright 2020 IBM
# Author: peter.zhong@au1.ibm.com
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the Apache 2.0 License.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Apache 2.0 License for more details.

from collections import deque

import Levenshtein
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from tqdm import tqdm


def _positive_span(value):
    """Return a valid HTML rowspan/colspan value."""
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1

class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)

    def get_tree_size(self):
        """Count total nodes in tree"""
        size = 1
        for child in self.children:
            size += child.get_tree_size()
        return size


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        denominator = self.maximum(*sequences)
        if denominator == 0:
            return 0.0
        return float(Levenshtein.distance(*sequences)) / denominator

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


class TEDS(object):
    """PubTabNet Tree-Edit-Distance-based Similarity."""

    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and n_jobs >= 1, 'n_jobs must be a positive integer'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = tuple(ignore_nodes or ())
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        # HTML headers have the same cell semantics as td for TEDS.
        tag = 'td' if node.tag == 'th' else node.tag
        if tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(tag,
                                 _positive_span(node.attrib.get('colspan', '1')),
                                 _positive_span(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            new_node = TableTree(tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if tag != 'td':
            for n in node.getchildren():
                if isinstance(n.tag, str):
                    self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def _extract_table(self, source):
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        try:
            root = html.fromstring(source, parser=parser)
        except (TypeError, ValueError, etree.ParserError):
            return None

        table = root if root.tag == 'table' else next(iter(root.xpath('.//table')), None)
        if table is None:
            return None
        if self.ignore_nodes:
            etree.strip_tags(table, *self.ignore_nodes)
        return table

    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        pred_table = self._extract_table(pred)
        true_table = self._extract_table(true)
        if pred_table is None or true_table is None:
            return 0.0

        pred_tree = self.load_html_tree(pred_table)
        true_tree = self.load_html_tree(true_table)
        
        tree_edit_distance = APTED(pred_tree, true_tree, CustomConfig()).compute_edit_distance()
        
        # Normalize to [0, 1] score using tree sizes
        pred_size = pred_tree.get_tree_size()
        true_size = true_tree.get_tree_size()
        max_size = max(pred_size, true_size)
        
        if max_size == 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - tree_edit_distance / max_size))

    def batch_evaluate(self, pred_json, true_json):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
            @params pred_json: {'FILENAME': 'HTML CODE', ...}
            @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
            @output: {'FILENAME': 'TEDS SCORE', ...}
        '''
        samples = true_json.keys()
        # if self.n_jobs == 1:
        scores = [self.evaluate(pred_json.get(filename, ''), true_json[filename]['html']) for filename in tqdm(samples)]
        # else:
        #     inputs = [{'pred': pred_json.get(filename, ''), 'true': true_json[filename]['html']} for filename in samples]
        #     scores = parallel_process(inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        scores = dict(zip(samples, scores))
        return scores


if __name__ == '__main__':
    import json
    import pprint
    with open('sample_pred.json') as fp:
        pred_json = json.load(fp)
    with open('sample_gt.json') as fp:
        true_json = json.load(fp)
    teds = TEDS(n_jobs=4)
    scores = teds.batch_evaluate(pred_json, true_json)
    pp = pprint.PrettyPrinter()
    pp.pprint(scores)
