while(true)​
{​
	if(!detail::call_all_input_node_accessors())​
		return;​

		detail::call_all_tasks​

	if(!detail::call_all_output_node_accessors(it->second))
		return;
}
