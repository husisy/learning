import spacy


def doc_basic_attribute(doc):
    str_fmt = '{:>15} {:>10} {:>10} {:>22} {:>10} {:>15} {:>10} {:>15}'
    print(str_fmt.format('text', 'dep_', 'head.text', 'children.text', 'n_lefts', 'lefts.text', 'n_rights', 'rights'))
    for x in doc:
        tmp1 = ' '.join(y.text for y in x.children)
        tmp2 = ' '.join(y.text for y in x.lefts)
        tmp3 = ' '.join(y.text for y in x.rights)
        print(str_fmt.format(x.text, x.dep_, x.head.text, tmp1, x.n_lefts, tmp2, x.n_rights, tmp3))
    root = [x for x in doc if x.head==x][0]
    subject = list(root.lefts)[0]
    print('root:: root.head==root: ', root.text)
    print('subject:: root.lefts[0]: ', subject.text)
    assert root.is_ancestor(subject)
    # doc[1].left_edge
    # doc[1].right_edge

def doc_tree_depth_first_search(doc):
    for x in doc:
        if len(list(x.children))==0:
            y = x
            path = [y.text]
            while y.head!=y:
                y = y.head
                path.append(y.text)
            print(path[::-1])

def displacy_serve_dep(doc):
    if input('type YES to confirm displacy.serve(dep): ')=='YES':
        spacy.displacy.serve(doc, style='dep')


def doc_parse_xcomp_comp(doc):
    for word in doc:
        if word.dep_ in ('xcomp', 'ccomp'):
            print(''.join(w.text_with_ws for w in word.subtree), '|', word.text)
            tmp1 = doc[word.left_edge.i: (word.right_edge.i+1)]
            print(tmp1.text, '|', tmp1.root.text)


if __name__=='__main__':
    nlp_sm = spacy.load('en_core_web_sm')
    doc1 = nlp_sm('Autonomous cars shift insurance liability toward manufacturers')
    doc2 = nlp_sm('Credit and mortgage account holders must submit their requests')
    doc3 = nlp_sm('displaCy uses CSS and JavaScript to show you how computers understand language')
    doc_basic_attribute(doc1)
    print()
    doc_tree_depth_first_search(doc1)
    print()
    displacy_serve_dep(doc1)
    doc_parse_xcomp_comp(doc3)
