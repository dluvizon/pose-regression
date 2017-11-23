# -*- coding: utf-8 -*-

class pa16j():
    """Alternated pose layout with 16 joints (like on Penn Action, but with
    three more joints on the spine.
    """
    num_joints = 16

    """Horizontal flip mapping"""
    map_hflip = [0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]

    """Projections from other layouts to the PA16J standard"""
    map_from_mpii = [6, 7, 8, 9, 12, 13, 11, 14, 10, 15, 2, 3, 1, 4, 0, 5]

    """Projections of PA16J to other formats"""
    map_to_mpii = [14, 12, 10, 11, 13, 15, 0, 1, 2, 3, 8, 6, 4, 5, 7, 9]

    """Color map"""
    color = ['g', 'r', 'b', 'y', 'm']
    cmap = [0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4]
    links = [[0, 1], [1, 2], [2, 3], [4, 6], [6, 8], [5, 7], [7, 9],
            [10, 12], [12, 14], [11, 13], [13, 15]]

