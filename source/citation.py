from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.util.docutils import SphinxDirective

class CitationDirective(SphinxDirective):
    """A directive for adding citation admonitions."""
    
    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {}

    def run(self):
        env = self.state.document.settings.env
        
        # Create an admonition node
        admonition_node = nodes.admonition()
        admonition_node['classes'] = ['admonition', 'citation']
        
        # Add title
        title_text = "Suggested Citation"
        textnodes, messages = self.state.inline_text(title_text, self.lineno)
        title = nodes.title(title_text, '', *textnodes)
        title['classes'] = ['admonition-title']
        admonition_node += title
        
        # Add content
        content = nodes.paragraph()
        self.state.nested_parse(self.content, self.content_offset, content)
        admonition_node += content
        
        return [admonition_node]

def setup(app):
    app.add_directive('citation', CitationDirective)
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }