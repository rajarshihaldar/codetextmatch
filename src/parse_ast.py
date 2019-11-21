import ast
import astpretty
import re



def get_node_str(node):
    # return str(node)[6:].split()[0]
    return str(type(node).__name__)

def process_ast(parent_node, node):
    if isinstance(node, str):
        return node

    if isinstance(node, ast.Module):
        bodies = node.body
        return_str = '('
        for body in bodies:
            return_str += ' '+process_ast(node,body)
        return_str += ' ) '
        return_str = re.sub(' +', ' ',return_str)
        return return_str

    if isinstance(node, ast.Num):
        # print(node.n)
        return 'Num ( '+str(node.n)+' ) '

    if isinstance(node, ast.Name):
        # print(node.id)
        return 'Name{} ( {} ) '.format(get_node_str(node.ctx),str(node.id))
    
    if isinstance(node, ast.Assign):
        return_str = 'Assign ( ( '
        for target in node.targets:
            return_str+='{} '.format(process_ast(node,target))
        return_str=return_str[:-1]+' ) ( {} ) ) '.format(process_ast(node,node.value))
        return return_str

    if isinstance(node, ast.Delete):
        return_str = 'Delete ( '
        for target in node.targets:
            return_str+=process_ast(node,target)+' '
        return_str=return_str[:-1]+' ) '
        return return_str

    if isinstance(node, ast.Starred):
        return 'Starred ( {} ) '.format(node.value)

    if isinstance(node, ast.UnaryOp):
        return_str = '{} ( {} ) '.format(process_ast(node,node.op),process_ast(node,node.operand))
        return return_str

    if isinstance(node, ast.USub) or isinstance(node, ast.Not) or isinstance(node, ast.Invert):
        return get_node_str(node)

    op_list = [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.MatMult, ast.UAdd, ast.USub, ast.Not, ast.Invert]
    for op in op_list:
        if isinstance(node, op):
            return get_node_str(node)


    if isinstance(node, ast.BinOp):
        # print(node.left)
        left_node = process_ast(node,node.left)
        right_node = process_ast(node,node.right)
        # print(right_node)
        oper = get_node_str(node.op)
        str2 = oper+' ( ' + left_node + ' ' + right_node + ' ) '
        return str2

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op,ast.And):
            return_str='And ( '
        if isinstance(node.op,ast.Or):
            return_str='Or ( '
        for val in node.values:
            return_str+=process_ast(node,val)+' '
        return_str=return_str[:-1]+' ) '
        return return_str

    if isinstance(node, ast.Compare):
        return_str = 'Compare ( '+process_ast(node,node.left)
        for op,comp in zip(node.ops,node.comparators):
            op_name = get_node_str(op)
            comp_name = process_ast(node,comp)
            return_str+=' '+op_name+' '+comp_name
        return '{} ) '.format(return_str)

    if isinstance(node, ast.Expr):
        val = node.value
        val = process_ast(node,val)
        return val

    if isinstance(node, ast.Call):
        func_name = ' ( '+process_ast(node,node.func)+' ) '
        # if isinstance(node.func, ast.Name):
        #     func_name = str(node.func.id)
        # elif isinstance(node.func, ast.Attribute):
        #     func_name = str(node.func.attr)
        # try:
        #     func_name = str(node.func.id)
        # except AttributeError:
        #     print("AttributeError")
        #     print(astpretty.pformat(node, show_offsets = False))
        #     exit()
        arg_str = ' ( '
        for args in node.args:
            arg_str += process_ast(node,args)
        arg_str+=' ) '
        try:
            return_str = 'Call ( '+func_name+' ( '+arg_str+' ) ) '
        except UnboundLocalError:
            print("UnboundLocalError")
            print(str(node.func))
            exit()
        return return_str

    if isinstance(node, ast.GeneratorExp):
        gen_elt = process_ast(node,node.elt)
        return_str = 'GeneratorExp ( '+gen_elt
        generators = ''
        for gen in node.generators:
            generators+=process_ast(node,gen)
        return_str+=' '+generators
        return_str=return_str+' ) '
        return return_str

    if isinstance(node, ast.ListComp):
        gen_elt = process_ast(node,node.elt)
        return_str = 'ListComp ( '+gen_elt
        generators = ''
        for gen in node.generators:
            generators+=process_ast(node,gen)
        return_str+=' '+generators
        return_str=return_str+' ) '
        return return_str

    if isinstance(node, ast.DictComp):
        gen_key = process_ast(node,node.key)
        gen_val = process_ast(node,node.value)
        return_str = 'DictComp ( '+gen_key+' '+gen_val
        generators = ''
        for gen in node.generators:
            generators+=process_ast(node,gen)
        return_str+=' '+generators
        return_str=return_str+' ) '
        
        return return_str

    if isinstance(node, ast.SetComp):
        gen_elt = process_ast(node,node.elt)
        return_str = 'SetComp ( '+gen_elt
        gen_elt = process_ast(node,node.elt)
        generators = ''
        for gen in node.generators:
            generators+=process_ast(node,gen)
            return_str+=' '+generators
        return_str=return_str+' ) '
        
        return return_str

    if isinstance(node, ast.comprehension):
        return_str = 'comprehension (target='+process_ast(node,node.target)+', iter='+process_ast(node,node.iter)
        return return_str

    if isinstance(node, ast.AnnAssign):
        return_str = 'AnnAssign (target='+process_ast(node,node.target)+', annotation='+process_ast(node,node.annotation)+', value='+process_ast(node,node.value)
        return return_str
    
    if isinstance(node, ast.AugAssign):
        return_str = 'AugAssign ( '+process_ast(node,node.target)+' '+process_ast(node,node.op)+' '+process_ast(node,node.value)
        return return_str

    if isinstance(node, ast.Raise):
        return_str = 'Raise ( '+process_ast(node,node.exc)+' '+process_ast(node,node.cause)
        return return_str

    if isinstance(node, ast.Assert):
        return_str = 'Assert ( '+process_ast(node,node.test)+' '+process_ast(node,node.msg)
        return return_str

    if isinstance(node, ast.Delete):
        return_str = 'Delete ('
        for target in node.targets:
            return_str+=' '+process_ast(node,target)
        return_str = return_str+' ) '
        return return_str

    if isinstance(node, ast.Pass):
        return 'Pass '

    if isinstance(node, ast.Str):
        val = 'Str ( '+str(node.s)+' ) '
        return val
    
    if isinstance(node, ast.JoinedStr):
        values = node.values
        return_str = ' ( '
        for value in values:
            return_str+=process_ast(node,value)
        return_str += ' ) '
        return return_str

    if isinstance(node, ast.FormattedValue):
        return 'FormattedValue ( '+process_ast(node,node.value)+' ) '

    if isinstance(node, ast.Bytes):
        return 'Bytes ( {} ) '.format(str(node.s))

    if isinstance(node, ast.List):
        return_str = 'List {} ( '.format(get_node_str(node.ctx))
        for val in node.elts:
            return_str+=process_ast(node,val)
        return_str+=' ) '
        return return_str

    if isinstance(node, ast.Tuple):
        return_str = 'Tuple {} ( '.format(get_node_str(node.ctx))
        for val in node.elts:
            return_str+=process_ast(node,val)
        return_str+=' ) '
        return return_str

    if isinstance(node, ast.Set):
        return_str = 'Set ( '
        for val in node.elts:
            return_str+=process_ast(node,val)
        return_str+=' ) '
        return return_str

    if isinstance(node, ast.Dict):
        keys = node.keys
        values = node.values
        return_str = 'Dict ( '
        for k,v in zip(keys,values):
            key = process_ast(node,k)
            val = process_ast(node,v)
            return_str+=' {} : {} '.format(key,val)
        return_str=return_str[:-1]+' ) '
        return return_str

    if isinstance(node, ast.Ellipsis):
        return 'Ellipsis '

    if isinstance(node, ast.NameConstant):
        return str(node.value)

    if isinstance(node, ast.keyword):
        return 'keyword ( {} {} ) '.format(process_ast(node,node.arg),process_ast(node,node.value))

    if isinstance(node, ast.IfExp):
        return 'IfExp ( {} {} {} ) '.format(process_ast(node,node.test),process_ast(node,node.body),process_ast(node,node.orelse))

    if isinstance(node, ast.Attribute):
        return 'Attribute ( {} {} ) '.format(process_ast(node,node.value),process_ast(node,node.attr))

    if isinstance(node, ast.Subscript):
        return 'Subscript ( {} {} {} ) '.format(process_ast(node,node.value),process_ast(node,node.slice),get_node_str(node.ctx))

    if isinstance(node, ast.Index):
        return 'Index ( {} ) '.format(process_ast(node,node.value))

    if isinstance(node, ast.Slice):
        return 'Slice ( {} {} {} ) '.format(process_ast(node,node.lower),process_ast(node,node.upper),process_ast(node,node.step))

    if isinstance(node, ast.ExtSlice):
        return_str = 'ExtSlice ('
        for dim in node.dims:
            return_str+=' '+process_ast(node,dim)
        return_str=return_str+' ) '
        return return_str

    if isinstance(node, ast.Import):
        return_str = 'Import ('
        for name in node.names:
            return_str+=' '+process_ast(node,name)
        return_str=return_str+' ) '
        return return_str

    if isinstance(node, ast.ImportFrom):
        return_str = 'ImportFrom ( '+str(node.module)
        for name in node.names:
            return_str+=' '+process_ast(node,name)+' '
        return_str=return_str+' ) '
        return return_str

    if isinstance(node, ast.alias):
        if node.asname:
            return 'alias ( {} {} ) '.format(node.name,node.asname)
        return 'alias ( {} ) '.format(node.name)

    if isinstance(node, ast.If):
        return_str = 'If( ( '+process_ast(node,node.test)+' ) ('
        for body in node.body:
            return_str+=' '+process_ast(node,body)
        return_str=return_str+' ) ('
        for orelse in node.orelse:
            return_str+=' '+process_ast(node,orelse)
        return_str=return_str+' ) ) '
        return return_str

    if isinstance(node, ast.For):
        return_str = 'For ( ( '+process_ast(node,node.target)+' ) ( '+process_ast(node,node.iter)+' ) ('
        for body in node.body:
            return_str+=' '+process_ast(node,body)
        return_str=return_str+' ) ('
        for orelse in node.orelse:
            return_str+=' '+process_ast(node,orelse)
        return_str=return_str+' ) ) '
        return return_str

    if isinstance(node, ast.While):
        return_str = 'While ( ( '+process_ast(node,node.test)+' ) ('
        for body in node.body:
            return_str+=' '+process_ast(node,body)
        return_str=return_str+' ) ('
        for orelse in node.orelse:
            return_str+=' '+process_ast(node,orelse)
        return_str=return_str+' ) ) '
        return return_str

    if isinstance(node, ast.Break):
        return 'Break '

    if isinstance(node, ast.Continue):
        return 'Continue '

    if isinstance(node, ast.Try):
        return_str = 'Try ( ( '
        for body in node.body:
            return_str+=process_ast(node,body)+' '
        return_str=return_str+' ) ( '
        # return_str+=process_ast(node,node.handlers)+' ) ( '
        for handler in node.handlers:
            return_str+=process_ast(node,handler)+' '
        return_str+=') ( '
        for orelse in node.orelse:
            return_str+=process_ast(node,orelse)+' '
        return_str=return_str+' ) ( '
        for finalbody in node.finalbody:
            return_str+=process_ast(node,finalbody)+' '
        return_str=return_str+' ) ) '
        return return_str

    if isinstance(node, ast.ExceptHandler):
        return_str = 'ExceptHandler ( ( '+process_ast(node,node.type)+' ) ( '+str(node.name)+' ) ('
        for body in node.body:
            return_str+=' '+process_ast(node,body)
        return_str=return_str+' ) ) '
        return return_str

    if isinstance(node, ast.With):
        return_str = 'With ( ('
        for items in node.items:
            return_str+=' '+process_ast(node,items)
        return_str=return_str+' ) ('
        for body in node.body:
            return_str+=' '+process_ast(node,body)
        return_str=return_str+' ) ) '
        return return_str

    if isinstance(node, ast.withitem):
        return_str = 'withitem ( ( '+process_ast(node,node.context_expr)+' ) ( '+process_ast(node,node.optional_vars)+' ) ) '
        return return_str

    if isinstance(node, ast.Return):
        return 'Return ( {} ) '.format(process_ast(node,node.value))

    if isinstance(node, ast.Yield):
        return 'Yield ( {} ) '.format(process_ast(node,node.value))

    if isinstance(node, ast.YieldFrom):
        return 'YieldFrom ( {} ) '.format(process_ast(node,node.value))

    if isinstance(node, ast.Global):
        return_str = ' '.join(node.names)
        return_str = 'Global ( '+return_str+' ) '
        return return_str

    if isinstance(node, ast.Nonlocal):
        return_str = ' '.join(node.names)
        return_str = 'Nonlocal ( '+return_str+' ) '
        return return_str

    if isinstance(node, ast.Await):
        return_str = 'Await ( {} ) '.format(process_ast(node,node.value))
        return return_str

    if isinstance(node, ast.AsyncFor):
        return_str = 'Async For ( ( '+process_ast(node,node.target)+' ) ( '+process_ast(node,node.iter)+' ) ('
        for body in node.body:
            return_str+=' '+process_ast(node,body)
        return_str=return_str+' ) ('
        for orelse in node.orelse:
            return_str+=' '+process_ast(node,orelse)
        return_str=return_str+' ) ) '
        return return_str

    if isinstance(node, ast.AsyncWith):
        return_str = 'Async With ( ('
        for items in node.items:
            return_str+=' '+process_ast(node,items)
        return_str=return_str+' ) ('
        for body in node.body:
            return_str+=' '+process_ast(node,body)
        return_str=return_str+' ) ) '
        return return_str
    
    # print(parent_node)
    # exit()
    
    if isinstance(node, ast.arguments):
        return_str = '('
        for arg in node.args:
            return_str += ' ' + arg.arg
        try:
            return_str += ' ' + node.vararg.arg
        except:
            pass
        try:
            for arg in node.kwonlyargs:
                return_str += ' ' + arg.arg
        except:
            pass
        try:
            return_str += ' ' + node.kwarg.arg
        except:
            pass
        return_str += ' ) '
        return return_str


    if isinstance(node, ast.FunctionDef):
        return_str = 'FunctionDef ( ( ' + node.name + ' ) ' + process_ast(node, node.args)
        return_str += 'Body ( '
        for body_elem in node.body:
            return_str += process_ast(node, body_elem)
        return_str += ' ) ) '
        return return_str

    if isinstance(node, ast.AsyncFunctionDef):
        return_str = 'Async FunctionDef ( ( ' + node.name + ' ) ' + process_ast(node, node.args)
        return_str += 'Body ( '
        for body_elem in node.body:
            return_str += process_ast(node, body_elem)
        return_str += ' ) ) '
        return return_str

    if isinstance(node, ast.Lambda):
        return_str = 'Lambda ( ' +  process_ast(node, node.args)
        return_str += 'Body ( ' + process_ast(node, node.body)
        return_str += ' ) ) '
        return return_str
    
    if isinstance(node, ast.ClassDef):
        return_str = 'ClassDef ( ( ' + node.name + ' ) '
        for base in node.bases:
            return_str += process_ast(node, base)
        for keyword in node.keywords:
            return_str += process_ast(node, keyword)
        return_str += 'Body ( '
        for body_elem in node.body:
            return_str += process_ast(node, body_elem)
        return_str += ' ) ) '
        # print(return_str)
        return return_str

    if node:
        print("Hello")
        print(node)
        print(astpretty.pformat(parent_node, show_offsets=False))
        exit()
        # print(node)
        try:
            return ast.dump(node)
        except TypeError:
            return_str=''
            for node_elem in node:
                return_str+=' ( '+ast.dump(node_elem)+' ) '
            return return_str
    else:
        return 'None'

    
if __name__ == "__main__":
    # tree_node = ast.parse('''a = 5''')
    # tree_node = ast.parse('''def f(a: 'annotation', b=1, c=2, *d, e, f=3, **g):\n\tpass''')
    tree_node = ast.parse('''class foo(base1, base2, metaclass=meta):\n\tpass''')
    print(astpretty.pformat(tree_node, show_offsets=False))
    s = process_ast(None, tree_node)
    print(s)
    exit()
