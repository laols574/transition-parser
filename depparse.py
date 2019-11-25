from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, Sequence, Text, Union

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

"""
Title: deparse.py
Author: Lauren Olson
This program acts as a transition dependency parser. It reads in text data, parses the input and 
uses an oracle to predict the transition which is used to train a classifer which then predicts 
transitions on more data.
"""

    
@dataclass()
class Dep:
    """A word in a dependency tree.
    The fields are defined by https://universaldependencies.org/format.html.
    """
    id: Text
    form: Union[Text, None]
    lemma: Union[Text, None]
    upos: Text
    xpos: Union[Text, None]
    feats: Sequence[Text]
    head: Union[Text, None]
    deprel: Union[Text, None]
    deps: Sequence[Text]
    misc: Union[Text, None]


def read_conllu(path: Text) -> Iterator[Sequence[Dep]]:
    """Reads a CoNLL-U format file into sequences of Dep objects.
    The CoNLL-U format is described in detail here:
    https://universaldependencies.org/format.html
    A few key highlights:
    * Word lines contain 10 fields separated by tab characters.
    * Blank lines mark sentence boundaries.
    * Comment lines start with hash (#).
    Each word line will be converted into a Dep object, and the words in a
    sentence will be collected into a sequence (e.g., list).
    :return: An iterator over sentences, where each sentence is a sequence of
    words, and each word is represented by a Dep object.
    """
    #read file into an array 
    with open(path) as f:
        file_array = f.readlines()
        
    array_of_lists = parse_input(file_array)
    
    for i in range(0, len(array_of_lists)):
        yield array_of_lists[i]
    return array_of_lists

def parse_input(file_array):
    array_of_sentences = []
    index = 0
    new_sentence = True
    for word in file_array:
        if(word == "\n"):
            index += 1
            new_sentence = True
            continue
        
        if(word[0] == "#"):
            continue
        
        d = initialize_fields(word)
        if(new_sentence):
            array_of_sentences.append([])
            array_of_sentences[index].append(d)
            new_sentence = False
        else:
            array_of_sentences[index].append(d)
    return array_of_sentences

      
def initialize_fields(word):
    word = word.split()
    id_num = word[0]
    if(word[0] == "_"):
        id_num = None
    form = word[1]
    if(word[1] == "_"):
        form = None
    lemma = word[2]
    if(word[2] == "_"):
        lemma = None
    upos = word[3]
    if(word[3] == "_"):
        upos = None
    xpos = word[4]
    if(word[4] == "_"):
        xpos = None
    
    feats = word[5]
    feats_arr = []
    for feat in feats.split("|"):
        if(feat != "_"):
            feats_arr.append(feat)
    feats = feats_arr
    
    head = word[6]
    if(word[6] == "_"):
        head = None
        
    deprel = word[7]
    if(word[7] == "_"):
        deprel = None
    
    deps = word[8]
    deps_arr = []
    for dep in deps.split("|"):
        if(dep != "_"):
            deps_arr.append(dep)
    deps = deps_arr
        
    misc = word[9]
    if(word[9] == "_"):
        misc = None
    return Dep(id_num, form, lemma, upos, xpos, feats, head, deprel, deps, misc)
        
class Action(Enum):
    """An action in an "arc standard" transition-based parser."""
    SHIFT = 1
    LEFT_ARC = 2
    RIGHT_ARC = 3


def parse(deps: Sequence[Dep],
          get_action: Callable[[Sequence[Dep], Sequence[Dep]], Action]) -> None:
    """Parse the sentence based on "arc standard" transitions.
    Following the "arc standard" approach to transition-based parsing, this
    method creates a stack and a queue, where the input Deps start out on the
    queue, are moved to the stack by SHIFT actions, and are combined in
    head-dependent relations by LEFT_ARC and RIGHT_ARC actions.
    This method does not determine which actions to take; those are provided by
    the `get_action` argument to the method. That method will be called whenever
    the parser needs a new action, and then the parser will perform whatever
    action is returned. If `get_action` returns an invalid action (e.g., a
    SHIFT when the queue is empty), an arbitrary valid action will be taken
    instead.
    This method does not return anything; it modifies the `.head` field of the
    Dep objects that were passed as input. Each Dep object's `.head` field is
    assigned the value of its head's `.id` field, or "0" if the Dep object is
    the root.
    :param deps: The sentence, a sequence of Dep objects, each representing one
    of the words in the sentence.
    :param get_action: a function or other callable that takes the parser's
    current stack and queue as input, and returns an "arc standard" action.
    :return: Nothing; the `.head` fields of the input Dep objects are modified.
    """
    #The future president joined the Guard in May 1968.
    #shift, shift, shift, left, left, shift, left, shift, shift, left, right, shift, shift, left, shift, right, right, shift, right
    stack = [] 
    queue = deps.copy()
    
    while(len(stack) > 1 or len(queue) > 0): #while state not final
        action = get_action(stack, queue)
        #move word from queue to stack 
        if(action == Action.SHIFT):
            if(len(queue) == 0):
                action = Action.LEFT_ARC
            else:
                stack.append(queue.pop(0))
            
        #top of the stack becomes the parent of the word below it
        #child.head = parent.id
        if(action == Action.LEFT_ARC):
            stack[len(stack) - 2].head = stack[len(stack) - 1].id
            stack.pop(len(stack) - 2)

        #top of the stack becomes child of word below it 
        #child.head = parent.id
        elif(action == Action.RIGHT_ARC):
            stack[len(stack) - 1].head = stack[len(stack) - 2].id
            stack.pop(len(stack) - 1)
    stack[0].head = "0"
    
class Oracle:
    def __init__(self, deps: Sequence[Dep]):
        """Initializes an Oracle to be used for the given sentence.
        Minimally, it initializes a member variable `actions`, a list that
        will be updated every time `__call__` is called and a new action is
        generated.
        Note: a new Oracle object should be created for each sentence; an
        Oracle object should not be re-used for multiple sentences.
        :param deps: The sentence, a sequence of Dep objects, each representing
        one of the words in the sentence.
        """
        self.deps = deps
        self.actions = []
        self.assigned = []
        self.features = []
        
        self.dep_map = {}
        for dep in self.deps:
            if(self.dep_map.get(dep.head) == None):
                self.dep_map[dep.head] = [dep]
            else: 
                self.dep_map[dep.head].append(dep)
                
    def __call__(self, stack: Sequence[Dep], queue: Sequence[Dep]) -> Action:
        """Returns the Oracle action for the given "arc standard" parser state.
        The oracle for an "arc standard" transition-based parser inspects the
        parser state and the reference parse (represented by the `.head` fields
        of the Dep objects) and:
        * Chooses LEFT_ARC if it produces a correct head-dependent relation
          given the reference parse and the current configuration.
        * Otherwise, chooses RIGHT_ARC if it produces a correct head-dependent
          relation given the reference parse and all of the dependents of the
          word at the top of the stack have already been assigned.
        * Otherwise, chooses SHIFT.
        The chosen action should be both:
        * Added to the `actions` member variable
        * Returned as the result of this method
        Note: this method should only be called on parser state based on the Dep
        objects that were passed to __init__; it should not be used for any
        other Dep objects.
        :param stack: The stack of the "arc standard" transition-based parser.
        :param queue: The queue of the "arc standard" transition-based parser.
        :return: The action that should be taken given the reference parse
        (the `.head` fields of the Dep objects).
        """
        #if the stack has two or less elements
        if( len(self.deps) - len(queue) <= 0):
            action = Action.SHIFT
            self.actions.append(action)
            temp_dict = add_features(stack , queue)
            self.features.append(temp_dict)
            return action
        else:
            stack_two  = stack[len(stack) - 2]
            stack_one = stack[len(stack) - 1]
            
            #LEFT ARC 
            #if it produces the correct head-dependent relation
            if(stack_two.head == stack_one.id):
                assigned_element = stack_two
                action = Action.LEFT_ARC
                self.assigned.append(assigned_element)
                
            #RIGHT ARC
            #produces the correct head-dependent relation 
            #all dependents of the top of the stack have already been assigned
            elif(stack_one.head == stack_two.id):

                #all dependents on the top of the stack have already been assigned
                #if the element on the top of the stack has no dependents
                if(self.dep_map.get(stack_one.id) == None):
                        action = Action.RIGHT_ARC
                        assigned_element = stack_one
                        self.assigned.append(assigned_element)

                else:
                    deps_check = []
                    #for all the dependents for the element at the top of the stack
                    for dep in self.dep_map[stack_one.id]:
                        #if its been assigned
                        if(dep in self.assigned):
                            deps_check.append(dep)
                    #if all of the dependents have been assigned
                    if(len(deps_check) == len(self.dep_map[stack_one.id])):
                        action = Action.RIGHT_ARC
                        assigned_element = stack_one
                        self.assigned.append(assigned_element)
                    else:
                        action = Action.SHIFT
                                            
            #move word from queue to stack 
            else:
                action = Action.SHIFT
        self.actions.append(action)
        #initialize features 
        temp_dict = add_features(stack , queue)
        #print("action: ", action, "features: ", temp_dict)
        self.features.append(temp_dict)
        return action
    
class Classifier:
    def __init__(self, parses: Iterator[Sequence[Dep]]):
        """Trains a classifier on the given parses.
        There are no restrictions on what kind of classifier may be trained,
        but a typical approach would be to
        1. Define features based on the stack and queue of an "arc standard"
           transition-based parser (e.g., part-of-speech tags of the top words
           in the stack and queue).
        2. Apply `Oracle` and `parse` to each parse in the input to generate
           training examples of parser states and oracle actions. It may be
           helpful to modify `Oracle` to call the feature extraction function
           defined in 1, and store the features alongside the actions list that
           `Oracle` is already creating.
        3. Train a machine learning model (e.g., logistic regression) on the
           resulting features and labels (actions).
        :param parses: An iterator over sentences, where each sentence is a
        sequence of words, and each word is represented by a Dep object.
        """
        self.clf = LogisticRegression(solver="liblinear", multi_class="ovr", dual=True)
        self.vectorizer = DictVectorizer()
        self.le = LabelEncoder()

        self.all_features = []
        self.all_labels = []
        
        for sentence in parses:
            oracle = Oracle(sentence)
            parse(sentence, oracle)
            self.all_features += oracle.features
            self.all_labels += oracle.actions
            
        feature_matrix = self.vectorizer.fit_transform(self.all_features)
        label_vector = self.le.fit_transform([label.value for label in self.all_labels])

        self.clf.fit(feature_matrix, label_vector)
                

    def __call__(self, stack: Sequence[Dep], queue: Sequence[Dep]) -> Action:
        """Predicts an action for the given "arc standard" parser state.
        There are no restrictions on how this prediction may be made, but a
        typical approach would be to convert the parser state into features,
        and then use the machine learning model (trained in `__init__`) to make
        the prediction.
        :param stack: The stack of the "arc standard" transition-based parser.
        :param queue: The queue of the "arc standard" transition-based parser.
        :return: The action that should be taken.
        """

        feats = add_features(stack, queue)
        new_features = self.vectorizer.transform(feats)
        
        return Action(self.le.inverse_transform(self.clf.predict(new_features))[0])
        
        
def add_features(stack, queue):
    #words, tags, n0, n, stack, parse
    stack_words = [word.form for word in stack]
    stack_tags = [word.upos for word in stack]
    
    queue_words = [word.form for word in queue]
    queue_tags = [word.upos for word in queue]
    features = {}
    stack1_form, stack2_form, stack3_form = get_stack_context(stack_words)
    stack1_pos, stack2_pos, stack3_pos = get_stack_context(stack_tags)
   
    queue1_form, queue2_form, queue3_form = get_queue_context(queue_words)
    queue1_pos, queue2_pos, queue3_pos = get_queue_context(queue_tags)
                    
    # Add word and tag unigrams
    index = 0
    for q in (queue1_form, queue2_form, queue3_form):
        if q:
            features['q%d=%s'  % (index, q)] = 1
            index += 1
    index = 0
    # Add word and tag unigrams
    for s in (stack1_form, stack2_form, stack3_form):
        if s:
            features['s%d=%s'  % (index, s)] = 1
            index += 1
    index = 0
    for t in (queue1_pos, queue2_pos, queue3_pos, stack1_pos, stack2_pos, stack3_pos):
        if t:
            features['t%d=%s' % (index, t)] = 1
            index += 1

    # Add word/tag pairs
    for i, (w, t) in enumerate(((queue1_form, queue1_pos), (queue2_form, queue2_pos), (queue3_form, queue3_pos), (stack1_form, stack1_pos))):
        if w or t:
            features['%d w=%s, t=%s' % (i, w, t)] = 1
    
    # Add some bigrams
    features['s0w=%s,  n0w=%s' % (stack1_form, queue1_form)] = 1
    features['wn0tn0-ws0 %s/%s %s' % (queue1_form, queue1_pos, stack1_form)] = 1
    features['wn0tn0-ts0 %s/%s %s' % (queue1_form, queue1_pos, stack1_pos)] = 1
    features['ws0ts0-wn0 %s/%s %s' % (stack1_form, stack1_pos, queue1_form)] = 1
    features['ws0-ts0 tn0 %s/%s %s' % (stack1_form, stack1_pos, queue1_pos)] = 1
    features['wt-wt %s/%s %s/%s' % (stack1_form, stack1_pos, queue1_form, queue1_pos)] = 1
    features['tt s0=%s n0=%s' % (stack1_pos, queue1_pos)] = 1
    features['tt n0=%s n1=%s' % (queue1_pos, queue2_pos)] = 1
    
    # Add some tag trigrams
    trigrams = ((queue1_pos, queue2_pos, queue3_pos), (stack1_pos, queue1_pos, queue2_pos), 
                (stack1_pos, stack2_pos, queue1_pos),(stack1_pos, stack2_pos, stack3_pos),
                (stack1_pos, stack2_pos, queue2_pos), (stack1_pos, stack2_pos, queue3_pos),
                (queue1_pos, queue2_pos, stack1_pos), (queue1_pos, queue2_pos, stack2_pos),
                (queue1_pos, queue2_pos, stack3_pos), (stack1_pos, queue1_pos, queue2_pos),
                )
    for i, (t1, t2, t3) in enumerate(trigrams):
        if t1 or t2 or t3:
            features['ttt-%d %s %s %s' % (i, t1, t2, t3)] = 1
    
    # Add some form trigrams
    trigrams = ((queue1_form, queue2_form, queue3_form), (stack1_form, queue1_form, queue2_form), 
                (stack1_form, stack2_form, queue1_form),(stack1_form, stack2_form, stack3_form),
                (stack1_form, stack2_form, queue2_form), (stack1_form, stack2_form, queue3_form),
                (queue1_form, queue2_form, stack1_form), (queue1_form, queue2_form, stack2_form),
                (queue1_form, queue2_form, stack3_form), (stack1_form, queue1_form, queue2_form),
                )
    for i, (t1, t2, t3) in enumerate(trigrams):
        if t1 or t2 or t3:
            features['ttt-%d %s %s %s' % (i, t1, t2, t3)] = 1

    # Add some mix grams
    trigrams = (
                (queue1_form, queue1_pos, queue2_pos, queue2_form, queue3_pos, queue3_form), 
                (stack1_form, stack1_pos, stack2_pos, stack2_form, stack3_pos, stack3_form),
                (queue1_form, queue1_pos, queue2_pos, queue2_form, stack1_pos, stack1_form), 
                (stack1_form, stack1_pos, stack2_pos, stack2_form, queue1_pos, queue1_form),
                (queue1_form, queue1_pos, stack1_pos, stack1_form, stack2_pos, stack2_form), 
                (stack1_form, stack1_pos, queue1_pos, queue1_form, queue2_pos, stack2_form), 
                )
    for i, (t1, t2, t3, t4, t5, t6) in enumerate(trigrams):
        if t1 or t2 or t3 or t4 or t5 or t6:
            features['ttt-%d %s %s %s' % (i, t1, t2, t3)] = 1
            
    features["len_stack"] = len(stack)
    features["len_queue"] = len(queue)
    return features

def get_stack_context(stack):
    stack1, stack2, stack3 = 0,0,0
    if(len(stack) > 0 and stack[-1] != None ):
        stack1 = stack[-1]
    if(len(stack) > 1 and stack[-2] != None):
        stack2 =  stack[-2]
    if(len(stack) > 2 and  stack[-3] != None):
        stack3 =  stack[-3]
    return stack1, stack2, stack3

def get_queue_context(queue):
    queue1, queue2, queue3 = 0,0,0
    if(len(queue) > 0 and queue[0] != None ):
        queue1 = queue[0]
    if(len(queue) > 1 and queue[1] != None):
        queue2 =  queue[1]
    if(len(queue) > 2 and  queue[2] != None):
        queue3 =  queue[2]
    return queue1, queue2, queue3
