import itertools
from typing import Sequence
import pytest

import depparse
from depparse import Dep, Action


def test_read_conllu():
    # read the entire training data and count the words and sentences read
    n_sentences = 0
    n_deps = 0
    for parse in depparse.read_conllu("UD_English-EWT/en_ewt-ud-train.conllu"):
        n_sentences += 1
        n_deps += len(parse)

        # make sure each sentence is a list of Dep objects
        assert all(isinstance(dep, depparse.Dep) for dep in parse)

        # make sure there is exactly one root node
        assert len([dep for dep in parse if dep.deprel == "root"]) == 1

    # make sure all sentences and words were read
    assert n_sentences == 12543
    assert n_deps == 204607

    # now do a deeper inspection of a single sentence from the training data

    # 1    Over    over    ADV    RB    _    2    advmod    2:advmod    _
    # 2    300    300    NUM    CD    NumType=Card    3    nummod    3:nummod    _
    # 3    Iraqis    Iraqis    PROPN    NNPS    Number=Plur    5    nsubj:pass    5:nsubj:pass|6:nsubj:xsubj|8:nsubj:pass    _
    # 4    are    be    AUX    VBP    Mood=Ind|Tense=Pres|VerbForm=Fin    5    aux:pass    5:aux:pass    _
    # 5    reported    report    VERB    VBN    Tense=Past|VerbForm=Part|Voice=Pass    0    root    0:root    _
    # 6    dead    dead    ADJ    JJ    Degree=Pos    5    xcomp    5:xcomp    _
    # 7    and    and    CCONJ    CC    _    8    cc    8:cc|8.1:cc    _
    # 8    500    500    NUM    CD    NumType=Card    5    conj    5:conj:and|8.1:nsubj:pass|9:nsubj:xsubj    _
    # 8.1    reported    report    VERB    VBN    Tense=Past|VerbForm=Part|Voice=Pass    _    _    5:conj:and    CopyOf=5
    # 9    wounded    wounded    ADJ    JJ    Degree=Pos    8    orphan    8.1:xcomp    _
    # 10    in    in    ADP    IN    _    11    case    11:case    _
    # 11    Fallujah    Fallujah    PROPN    NNP    Number=Sing    5    obl    5:obl:in    _
    # 12    alone    alone    ADV    RB    _    11    advmod    11:advmod    SpaceAfter=No
    # 13    .    .    PUNCT    .    _    5    punct    5:punct    _
    parses = depparse.read_conllu("UD_English-EWT/en_ewt-ud-train.conllu")
    [parse] = itertools.islice(parses, 61, 62)
    assert parse[0] == depparse.Dep(
        "1", "Over", "over", "ADV", "RB", [], "2", "advmod", ["2:advmod"], None)
    assert parse[2] == depparse.Dep(
        "3", "Iraqis", "Iraqis", "PROPN", "NNPS", ["Number=Plur"], "5",
        "nsubj:pass", ["5:nsubj:pass", "6:nsubj:xsubj", "8:nsubj:pass"], None)
    assert parse[3] == depparse.Dep(
        "4", "are", "be", "AUX", "VBP",
        ["Mood=Ind", "Tense=Pres", "VerbForm=Fin"],
        "5", "aux:pass", ["5:aux:pass"], None)
    assert parse[4] == depparse.Dep(
        "5", "reported", "report", "VERB", "VBN",
        ["Tense=Past", "VerbForm=Part", "Voice=Pass"],
        "0", "root", ["0:root"], None)
    assert parse[8] == depparse.Dep(
        "8.1", "reported", "report", "VERB", "VBN",
        ["Tense=Past", "VerbForm=Part", "Voice=Pass"],
        None, None,    ["5:conj:and"], "CopyOf=5")


def test_parse():
    # consider a specific sentence from the training data

    # # sent_id = weblog-blogspot.com_alaindewitt_20040929103700_ENG_20040929_103700-0026
    # # text = The future president joined the Guard in May 1968.
    # 1    The    the    DET    DT    Definite=Def|PronType=Art    3    det    3:det    _
    # 2    future    future    ADJ    JJ    Degree=Pos    3    amod    3:amod    _
    # 3    president    president    NOUN    NN    Number=Sing    4    nsubj    4:nsubj    _
    # 4    joined    join    VERB    VBD    Mood=Ind|Tense=Past|VerbForm=Fin    0    root    0:root    _
    # 5    the    the    DET    DT    Definite=Def|PronType=Art    6    det    6:det    _
    # 6    Guard    Guard    PROPN    NNP    Number=Sing    4    obj    4:obj    _
    # 7    in    in    ADP    IN    _    8    case    8:case    _
    # 8    May    May    PROPN    NNP    Number=Sing    4    obl    4:obl:in    _
    # 9    1968    1968    NUM    CD    NumType=Card    8    nummod    8:nummod    SpaceAfter=No
    # 10    .    .    PUNCT    .    _    4    punct    4:punct    _
    parses = depparse.read_conllu("UD_English-EWT/en_ewt-ud-train.conllu")
    [deps] = itertools.islice(parses, 352, 353)

    # clear out all the head information
    orig_heads = clear_heads(deps)

    # run the parser with the oracle list of actions
    depparse.parse(deps, IterActions([
        Action.SHIFT,
        Action.SHIFT,
        Action.SHIFT,
        Action.LEFT_ARC,
        Action.LEFT_ARC,
        Action.SHIFT,
        Action.LEFT_ARC,
        Action.SHIFT,
        Action.SHIFT,
        Action.LEFT_ARC,
        Action.RIGHT_ARC,
        Action.SHIFT,
        Action.SHIFT,
        Action.LEFT_ARC,
        Action.SHIFT,
        Action.RIGHT_ARC,
        Action.RIGHT_ARC,
        Action.SHIFT,
        Action.RIGHT_ARC,
    ]))
    
    # make sure that the original heads have been restored by the parser
    assert [dep.head for dep in deps] == orig_heads


def test_oracle():
    # consider a specific sentence from the training data

    # # sent_id = answers-20111108085734AATXy0E_ans-0004
    # # text = Plaster of Paris does two things
    # 1    Plaster    plaster    NOUN    NN    Number=Sing    4    nsubj    4:nsubj    _
    # 2    of    of    ADP    IN    _    3    case    3:case    _
    # 3    Paris    Paris    PROPN    NNP    Number=Sing    1    nmod    1:nmod:of    _
    # 4    does    do    VERB    VBZ    Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin    0    root    0:root    _
    # 5    two    two    NUM    CD    NumType=Card    6    nummod    6:nummod    _
    # 6    things    thing    NOUN    NNS    Number=Plur    4    obj    4:obj    _
    parses = depparse.read_conllu("UD_English-EWT/en_ewt-ud-train.conllu")
    [deps] = itertools.islice(parses, 7475, 7476)

    # create an oracle for the sentence and try a few actions
    oracle = depparse.Oracle(deps)
    # shift on an empty stack
    assert oracle([], deps) == Action.SHIFT
    # shift on a stack with only one entry
    assert oracle(deps[:1], deps[1:]) == Action.SHIFT
    # shift because "Plaster" and "of" are not in a head-dependent relation
    assert oracle(deps[:2], deps[2:]) == Action.SHIFT
    # left-arc because "Paris" is the head of "of"
    assert oracle(deps[:3], deps[3:]) == Action.LEFT_ARC
    # right-arc because "Plaster" is the head of "Paris"
    assert oracle(deps[:1] + deps[2:3], deps[3:]) == Action.RIGHT_ARC

    # create a new oracle for the same sentence and extract all the actions
    oracle = depparse.Oracle(deps)
    depparse.parse(deps, oracle)

    assert oracle.actions == [
        Action.SHIFT, #1
        Action.SHIFT, #2
        Action.SHIFT,#3
        Action.LEFT_ARC,#4
        Action.RIGHT_ARC,#5
        Action.SHIFT,#6
        Action.LEFT_ARC,#7
        Action.SHIFT,#8
        Action.SHIFT,#9
        Action.LEFT_ARC,#10
        Action.RIGHT_ARC,#11
    ]
    '''print([(dep.form, dep.head, dep.id) for dep in deps])
    print(oracle.actions)
    for i in range(0, len(oracle.features)):
        print("action:", oracle.actions[i], "features", oracle.features[i])'''

def test_oracle_round_trip():
    # take the first 50 parses from the training data
    parses = depparse.read_conllu("UD_English-EWT/en_ewt-ud-train.conllu")
    for i, deps in enumerate(itertools.islice(parses, 50)):

        # skip the non-projective parses
        if i in {4, 21, 25, 31}:
            continue

        # collect the head for each word
        orig_heads = [dep.head for dep in deps]

        # run the oracle to determine the sequence of actions
        oracle = depparse.Oracle(deps)
        depparse.parse(deps, oracle)

        # clear out all the head information
        clear_heads(deps)

        # feed the oracle-identified actions in, one at a time
        depparse.parse(deps, IterActions(oracle.actions))
        
        # make sure that the original heads have been restored by the parser
        assert [dep.head for dep in deps] == orig_heads


@pytest.fixture(scope="module")
def full_model_accuracy():
    # train a classifier on the entire training data
    train_parses = depparse.read_conllu("UD_English-EWT/en_ewt-ud-train.conllu")
    classifier = depparse.Classifier(train_parses)

    # test the classifier on the development set
    correct = 0
    total = 0
    for deps in depparse.read_conllu("UD_English-EWT/en_ewt-ud-dev.conllu"):
        total += len(deps)

        # clear out all the head information
        orig_heads = clear_heads(deps)

        # parse using the classifier to predict actions
        depparse.parse(deps, classifier)

        # count how many of the heads have been correctly restored
        for dep, orig_head in zip(deps, orig_heads):
            if dep.head == orig_head:
                correct += 1

    # return the accuracy
    return correct / total


def test_accuracy(full_model_accuracy, capsys):
    if capsys is not None:
        with capsys.disabled():
            msg = "\n{:.1%} accuracy on EWT development data"
            print(msg.format(full_model_accuracy))
    assert full_model_accuracy >= 0.625


@pytest.mark.xfail
def test_very_accurate_prediction(full_model_accuracy):
    assert full_model_accuracy >= 0.75


class IterActions:
    """A class for feeding a list of actions, one at a time, to a parser"""

    def __init__(self, actions: Sequence[Action]):
        self.actions_iter = iter(actions)

    def __call__(self, stack: Sequence[Dep], queue: Sequence[Dep]):
        return next(self.actions_iter)


def clear_heads(deps: Sequence[Dep]):
    """Removes all head information (.head, .deprel, .deps) from the sentence"""
    heads = []
    for dep in deps:
        heads.append(dep.head)
        dep.head = None
        dep.deprel = None
        dep.deps = None
    return heads
